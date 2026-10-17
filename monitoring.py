"""
monitoring.py - Release checks for the NT assault forecasting system
(Assessment 4)

Run order:
    python Updated_End_to_End_pipeline.py     # panel, Model A, Outputs/R*
    python evaluation.py                      # backtest, Outputs/E2-E5
    python forecast.py                        # forecasts, Outputs/F1, E6
    python monitoring.py                      # this file, Outputs/M1-M3

This script implements the team's monitoring design
(documents/Monitoring_Design_DAN2_Theme2.docx). It is run after each new NT
crime statistics release and answers the design's three questions:

    1. Is the input data complete and timely?          (D checks)
    2. Are forecasts still accurate enough to plan with? (P checks)
    3. Has the pattern of assaults shifted?              (R checks)

Every check returns PASS, WARN or FAIL:
    PASS  no action
    WARN  document it and watch the next release
    FAIL  do not promote the new forecasts to the Administrator Portal;
          open an investigation; retrain only if the FAIL repeats or is
          explained by lasting drift

Thresholds (from the monitoring design, section 2.2)
----------------------------------------------------
h = 1 (XGBoost)            WARN if skill vs seasonal naive < 10%,
                           FAIL if skill < 0%, over the last 6 scored origins
h = 3, 6, 12 (benchmark)   WARN if RMSE > 1.25 x backtest RMSE,
                           FAIL if skill vs seasonal naive < 0%
80% intervals              WARN if realised coverage is outside 70-90%
                           over the last 12 scored months
Provisional actuals        never FAIL: a FAIL on months still being
                           recorded is reported as WARN ([A9], [A23])

[A24] Drift thresholds were not fixed in the design, which leaves drift to
      human review. This script uses initial values: WARN if a region's mean
      rate over the latest settled quarter differs from the same quarter of
      the previous three years by more than 25%, or if the alcohol- or
      DV-involved share of assaults moves by more than 5 percentage points.
      A drift WARN never triggers retraining by itself.

Outputs
-------
Outputs/M1_monitoring_checks.csv     one row per check, with value, rule and status
Outputs/M2_monitoring_scored.csv     every scored forecast: month, region, horizon,
                                     method, actual, forecast, error, interval flag
Outputs/M3_monitoring_summary.json   overall status and the two decisions:
                                     promote the forecasts? retrain the models?

Nothing here refits a model or changes an earlier output. The check date is
recorded so a later run can be compared with this one.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import evaluation as ev
import forecast as fc
import Updated_End_to_End_pipeline as pl

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "Outputs"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
RANK = {PASS: 0, WARN: 1, FAIL: 2}

# Monitoring design, section 2.2
H1_WARN_SKILL = 10.0          # %
H1_FAIL_SKILL = 0.0           # %
H1_ROLLING_ORIGINS = 6
LONG_RMSE_RATIO = 1.25
COVERAGE_BAND = (70.0, 90.0)  # % for a nominal 80% interval
COVERAGE_MONTHS = 12
LONG_ROLLING_MONTHS = 12

# [A24] initial drift thresholds
DRIFT_RATE_CHANGE = 0.25      # 25% change in a region's quarterly mean rate
DRIFT_SHARE_PP = 5.0          # percentage points
FRESHNESS_DAYS = 62           # NT crime statistics are published monthly

KNOWN_GAPS = {pd.Timestamp("2023-11-01")}   # PROMIS -> SerPro, documented [A3]


class Log:
    """Collects check results in the order they run."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, check_id: str, area: str, check: str, value, rule: str,
            status: str, note: str = "") -> None:
        self.rows.append({"Check_id": check_id, "Area": area, "Check": check,
                          "Value": value, "Rule": rule, "Status": status,
                          "Note": note})
        print(f"  {status:<4}  {check_id:<4} {check}: {value}" + (f"  ({note})" if note else ""))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def worst(statuses) -> str:
    statuses = list(statuses)
    return max(statuses, key=RANK.get) if statuses else PASS


def cap_provisional(status: str) -> str:
    """The design never scores FAIL on provisional actuals."""
    return WARN if status == FAIL else status


def rmse(err: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def skill(model_rmse: float, naive_rmse: float) -> float:
    return float((1 - model_rmse / naive_rmse) * 100)


# ---------------------------------------------------------------------------
# D: data quality and freshness (design 2.1)
# ---------------------------------------------------------------------------

def data_checks(log: Log, monthly: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    print("\n[monitor] D - data quality and freshness")

    missing = []
    for spec in pl.EXPECTED_FILES.values():
        try:
            pl.find_source_file(spec["filename"])
        except FileNotFoundError:
            missing.append(spec["filename"])
    log.add("D1", "Data", "Source files present",
            f"{len(pl.EXPECTED_FILES) - len(missing)} of {len(pl.EXPECTED_FILES)}",
            "all expected files resolve in dataset/source/",
            FAIL if missing else PASS, ", ".join(missing))

    raw_new = pd.read_csv(pl.RAW_DIR / "crime_new.csv", nrows=5)
    raw_cols = list(raw_new.columns)
    canonical = {c.strip().lower(): c for c in pl.KEEP_COLS}
    absent = [c for k, c in canonical.items()
              if k not in {r.strip().lower() for r in raw_cols}]
    drifted = [r for r in raw_cols
               if r.strip().lower() in canonical and r not in pl.KEEP_COLS]
    log.add("D2", "Data", "Crime file schema",
            "all required columns" if not absent else f"missing {absent}",
            "required columns present; spacing or case drift is handled",
            FAIL if absent else PASS,
            ("header drift handled: " + ", ".join(repr(d) for d in drifted)) if drifted else "")

    old = pl._read_raw("crime_old")
    old = old[old["Year"].between(pl.YEAR_MIN, pl.YEAR_MAX)]
    unmapped = sorted(set(old["Offence category"].dropna()) - set(pl.CATEGORY_MAP))
    new = pl._read_raw("crime_new")
    has_assault = pl.ASSAULT_CATEGORY in set(new["Offence category"].dropna())
    log.add("D3", "Data", "Offence categories mapped",
            f"{len(unmapped)} unmapped; '{pl.ASSAULT_CATEGORY}' present: {has_assault}",
            "no unmapped PROMIS category; ANZSOC assault category present",
            FAIL if (unmapped or not has_assault) else PASS, ", ".join(unmapped))

    region_col = "Reporting Region" if "Reporting Region" in new else "Reporting region"
    balance = new[new[region_col] == "NT Balance"]
    sa2 = balance["Statistical Area 2"].dropna().astype(str).str.strip()
    no_location = int((sa2 == "Unknown").sum())
    unknown_sa2 = sorted(set(sa2) - set(pl.SA2_TO_REGION) - {"Unknown"})
    unknown_regions = sorted(set(new[region_col].dropna())
                             - set(pl.REGION_TO_POP) - {"Unknown"})
    log.add("D4", "Data", "Region crosswalk covers all rows",
            f"{len(unknown_regions)} new police region(s), {len(unknown_sa2)} new SA2 name(s)",
            "every police region and NT Balance SA2 maps to an NTG region",
            WARN if (unknown_regions or unknown_sa2) else PASS,
            ", ".join(unknown_regions + unknown_sa2)
            or f"{no_location} NT Balance rows have SA2 'Unknown' and go to Top End [A5]")

    grid = pd.MultiIndex.from_product(
        [sorted(monthly["Region"].unique()),
         pd.date_range(monthly["date"].min(), monthly["date"].max(), freq="MS")],
        names=["Region", "date"]).to_frame(index=False)
    have = grid.merge(monthly[["Region", "date"]], on=["Region", "date"],
                      how="left", indicator=True)
    gaps = sorted(set(have.loc[have["_merge"] == "left_only", "date"]))
    unexplained = [g for g in gaps if g not in KNOWN_GAPS]
    log.add("D5", "Data", "Calendar completeness",
            f"{len(gaps)} missing month(s): " + (", ".join(f"{g:%Y-%m}" for g in gaps) or "none"),
            "only documented gaps; gaps stay NaN, never imputed",
            WARN if unexplained else PASS,
            "Nov 2023 is the documented PROMIS-SerPro gap [A3]" if gaps and not unexplained else "")

    try:
        year = fc.check_reconstruction(new, monthly)
        log.add("D6", "Data", "Raw file rebuilds the processed panel",
                f"{year} matches exactly", "rebuilt region-month counts equal the panel", PASS)
    except ValueError as exc:
        log.add("D6", "Data", "Raw file rebuilds the processed panel",
                "mismatch", "rebuilt region-month counts equal the panel", FAIL, str(exc))

    as_at = pd.to_datetime(str(new["As At"].iloc[0]), dayfirst=True)
    crime_dates = pd.to_datetime(dict(year=new["Year"], month=new["Month number"], day=1))
    latest_crime = crime_dates.max()
    settled_until = as_at - pd.DateOffset(months=ev.PROVISIONAL_MONTHS)
    provisional = sorted(d for d in crime_dates.unique() if d > settled_until)
    log.add("D7", "Data", "Provisional window flagged",
            f"{len(provisional)} provisional month(s): "
            f"{provisional[0]:%Y-%m} to {provisional[-1]:%Y-%m}" if provisional else "none",
            f"months within {ev.PROVISIONAL_MONTHS} months of the As At date are not scored as FAIL",
            PASS, f"crime file As At {as_at:%d %b %Y}")

    age = as_of - as_at
    log.add("D8", "Data", "Crime release freshness",
            f"As At {as_at:%d %b %Y}, {age.days} days before the check date",
            f"WARN if older than {FRESHNESS_DAYS} days (monthly release)",
            WARN if age.days > FRESHNESS_DAYS else PASS,
            "check data.nt.gov.au for a newer release" if age.days > FRESHNESS_DAYS else "")

    pac = monthly.loc[monthly["Total_PAC"].notna(), "date"]
    pac_last = pac.max() if not pac.empty else pd.NaT
    lag_q = ((latest_crime.year - pac_last.year) * 4
             + (latest_crime.quarter - pac_last.quarter)) if pd.notna(pac_last) else np.nan
    log.add("D9", "Data", "Alcohol supply lag",
            f"latest supply quarter {pac_last:%Y}-Q{pac_last.quarter}, "
            f"{lag_q} quarter(s) behind the latest crime month" if pd.notna(pac_last) else "no supply data",
            "WARN if more than 2 quarters behind; Model B only",
            WARN if (pd.isna(lag_q) or lag_q > 2) else PASS,
            "supply is context only; the forecasts do not use it")

    pop_year = int(monthly["Year"].max())
    log.add("D10", "Data", "Population estimates cover the crime months",
            f"population to {pop_year}, crime to {latest_crime:%Y-%m}",
            "WARN if crime months run past the latest population year",
            WARN if latest_crime.year > pop_year else PASS,
            f"{latest_crime.year} rates use {pop_year} population [A22]; "
            f"the panel waits for new estimates [A1]" if latest_crime.year > pop_year else "")

    return {"as_at": as_at, "settled_until": settled_until,
            "latest_crime": latest_crime, "provisional": provisional}


# ---------------------------------------------------------------------------
# P: forecast performance (design 2.2)
# ---------------------------------------------------------------------------

def coverage_out_of_sample(sub: pd.DataFrame, months: list) -> tuple[float, int]:
    """Coverage of the 80% interval for the given target months, each built
    only from residuals of earlier target months (as in evaluation.py E5)."""
    hits, checked = 0, 0
    for month in months:
        past = sub.loc[sub["target"] < month, "residual"]
        if len(past) < 12 * 6:
            continue
        lo, hi = past.quantile([0.10, 0.90])
        now = sub[sub["target"] == month]
        hits += int(((now["residual"] >= lo) & (now["residual"] <= hi)).sum())
        checked += len(now)
    return (hits / checked * 100 if checked else np.nan), checked


def performance_checks(log: Log) -> tuple[pd.DataFrame, dict]:
    print("\n[monitor] P - forecast performance")
    winners, intervals = fc.recommended_by_horizon()
    backtest_rmse = (pd.read_csv(OUT_DIR / "E4_backtest_metrics.csv")
                     .set_index(["Model", "Horizon_months"])["RMSE"])

    bt = pd.read_csv(OUT_DIR / "E4_backtest_predictions.csv",
                     parse_dates=["origin", "target"])
    bt["error"] = bt["pred"] - bt["actual"]
    bt["residual"] = -bt["error"]
    bt["actual_status"] = "settled"

    scored = []
    settled_status = {}

    for h in ev.HORIZONS:
        model = winners[h]
        sub = bt[(bt["h"] == h) & (bt["Model"] == model)].copy()
        naive = bt[(bt["h"] == h) & (bt["Model"] == ev.SEASONAL_NAIVE)]

        if h == 1:
            last_origins = sorted(sub["origin"].unique())[-H1_ROLLING_ORIGINS:]
            window = sub[sub["origin"].isin(last_origins)]
            n_window = naive[naive["origin"].isin(last_origins)]
            s = skill(rmse(window["error"]), rmse(n_window["error"]))
            status = FAIL if s < H1_FAIL_SKILL else (WARN if s < H1_WARN_SKILL else PASS)
            log.add("P1", "Performance", f"h=1 {model}: skill, last {H1_ROLLING_ORIGINS} settled origins",
                    f"{s:.1f}% (targets {window['target'].min():%Y-%m} to {window['target'].max():%Y-%m})",
                    f"WARN < {H1_WARN_SKILL:.0f}%, FAIL < {H1_FAIL_SKILL:.0f}%", status)
        else:
            last_months = sorted(sub["target"].unique())[-LONG_ROLLING_MONTHS:]
            window = sub[sub["target"].isin(last_months)]
            n_window = naive[naive["target"].isin(last_months)]
            r = rmse(window["error"])
            ratio = r / float(backtest_rmse[(model, h)])
            s = skill(r, rmse(n_window["error"]))
            fails = model != ev.SEASONAL_NAIVE and s < 0
            status = FAIL if fails else (WARN if ratio > LONG_RMSE_RATIO else PASS)
            log.add(f"P{h}" if h < 10 else "P12", "Performance",
                    f"h={h} {model}: last {LONG_ROLLING_MONTHS} settled months",
                    f"RMSE {r:.1f} = {ratio:.2f} x backtest; skill {s:.1f}%",
                    f"WARN if > {LONG_RMSE_RATIO} x backtest RMSE, FAIL if skill < 0%",
                    status, "skill is 0 by definition for the seasonal naive"
                    if model == ev.SEASONAL_NAIVE else "")
        settled_status[h] = status

        cov, n = coverage_out_of_sample(sub, sorted(sub["target"].unique())[-COVERAGE_MONTHS:])
        c_status = WARN if (np.isnan(cov) or not COVERAGE_BAND[0] <= cov <= COVERAGE_BAND[1]) else PASS
        log.add(f"C{h}", "Performance", f"h={h} 80% interval coverage, last {COVERAGE_MONTHS} settled months",
                f"{cov:.1f}% of {n}", f"WARN outside {COVERAGE_BAND[0]:.0f}-{COVERAGE_BAND[1]:.0f}%",
                c_status)

        lo = float(intervals.loc[h, "Lower_offset"])
        hi = float(intervals.loc[h, "Upper_offset"])
        recent = sub[sub["origin"].isin(sorted(sub["origin"].unique())[-H1_ROLLING_ORIGINS:])].copy()
        recent["inside_80"] = (recent["residual"] >= lo) & (recent["residual"] <= hi)
        scored.append(recent)

    # Newest months: frozen models on the 2026 releases (forecast.py E6)
    e6_path = OUT_DIR / "E6_holdout_2026_predictions.csv"
    if e6_path.exists():
        e6 = pd.read_csv(e6_path, parse_dates=["origin", "target"])
        e6["error"] = e6["pred"] - e6["actual"]
        e6["residual"] = -e6["error"]
        for h in sorted(e6["h"].unique()):
            model = winners[h]
            sub = e6[(e6["h"] == h) & (e6["Model"] == model)].copy()
            naive = e6[(e6["h"] == h) & (e6["Model"] == ev.SEASONAL_NAIVE)]
            r = rmse(sub["error"])
            s = skill(r, rmse(naive["error"]))
            if h == 1:
                status = FAIL if s < H1_FAIL_SKILL else (WARN if s < H1_WARN_SKILL else PASS)
                rule = f"WARN < {H1_WARN_SKILL:.0f}%, FAIL < {H1_FAIL_SKILL:.0f}% (FAIL shown as WARN)"
            else:
                ratio = r / float(backtest_rmse[(model, h)])
                status = FAIL if s < 0 else (WARN if ratio > LONG_RMSE_RATIO else PASS)
                rule = f"WARN if > {LONG_RMSE_RATIO} x backtest RMSE, FAIL if skill < 0% (FAIL shown as WARN)"
            raw = status
            status = cap_provisional(status)
            months = f"{sub['target'].min():%Y-%m} to {sub['target'].max():%Y-%m}"
            log.add(f"N{h}", "Performance", f"h={h} {model} on newest months (provisional)",
                    f"RMSE {r:.1f}, skill {s:.1f}%, bias {sub['error'].mean():+.1f} ({months})",
                    rule, status, "actuals still being recorded [A23]"
                    + (f"; raw result {raw}" if raw != status else ""))
            lo = float(intervals.loc[h, "Lower_offset"])
            hi = float(intervals.loc[h, "Upper_offset"])
            sub["inside_80"] = (sub["residual"] >= lo) & (sub["residual"] <= hi)
            cov = float(sub["inside_80"].mean() * 100)
            c_status = cap_provisional(WARN if not COVERAGE_BAND[0] <= cov <= COVERAGE_BAND[1] else PASS)
            log.add(f"NC{h}", "Performance", f"h={h} 80% interval coverage on newest months (provisional)",
                    f"{cov:.1f}% of {len(sub)}",
                    f"WARN outside {COVERAGE_BAND[0]:.0f}-{COVERAGE_BAND[1]:.0f}%", c_status,
                    "too few months to judge; re-score when final")
            sub["actual_status"] = "provisional"
            scored.append(sub)

    out = pd.concat(scored, ignore_index=True)
    out = out.rename(columns={"target": "Target_month", "origin": "Origin", "h": "Horizon_months",
                              "pred": "Forecast_rate_100k", "actual": "Actual_rate_100k",
                              "error": "Error_forecast_minus_actual",
                              "actual_status": "Actual_status", "inside_80": "Inside_80_interval"})
    out["Origin"] = pd.to_datetime(out["Origin"]).dt.strftime("%Y-%m")
    out["Target_month"] = pd.to_datetime(out["Target_month"]).dt.strftime("%Y-%m")
    out = out[["Target_month", "Region", "Horizon_months", "Model", "Origin",
               "Actual_rate_100k", "Forecast_rate_100k", "Error_forecast_minus_actual",
               "Inside_80_interval", "Actual_status"]]
    out = out.sort_values(["Actual_status", "Horizon_months", "Target_month", "Region"],
                          ascending=[False, True, True, True]).round(1)
    return out, settled_status


# ---------------------------------------------------------------------------
# R: drift in the pattern of assaults (design 2.3)
# ---------------------------------------------------------------------------

def drift_checks(log: Log, monthly: pd.DataFrame, settled_until: pd.Timestamp) -> None:
    print("\n[monitor] R - drift (human review, never an automatic retrain)")
    settled = monthly[monthly["date"] <= settled_until]
    last = settled["date"].max()
    quarter = pd.date_range(last - pd.DateOffset(months=2), last, freq="MS")

    shifted = []
    for region, grp in settled.groupby("Region"):
        now = grp.loc[grp["date"].isin(quarter), "Assault_rate_100k"].mean()
        past = grp.loc[grp["date"].isin(
            [d - pd.DateOffset(years=k) for d in quarter for k in (1, 2, 3)]), "Assault_rate_100k"]
        change = now / past.mean() - 1
        if abs(change) > DRIFT_RATE_CHANGE:
            shifted.append(f"{region} {change:+.0%}")
    log.add("R1", "Drift", "Regional rate vs same quarter, previous 3 years",
            f"{len(shifted)} region(s) shifted ({quarter[0]:%b}-{quarter[-1]:%b %Y})",
            f"WARN if any region changes by more than {DRIFT_RATE_CHANGE:.0%} [A24]",
            WARN if shifted else PASS, ", ".join(shifted))

    year_now = settled[settled["date"] > last - pd.DateOffset(months=12)]
    year_past = settled[(settled["date"] <= last - pd.DateOffset(months=12))
                        & (settled["date"] > last - pd.DateOffset(months=48))]
    for check_id, col, label in [("R2", "Alcohol_offences", "Alcohol-involved"),
                                 ("R3", "DV_offences", "DV-involved")]:
        now = year_now[col].sum() / year_now["Assault_offences"].sum() * 100
        past = year_past[col].sum() / year_past["Assault_offences"].sum() * 100
        diff = now - past
        log.add(check_id, "Drift", f"{label} share of assaults, last 12 settled months",
                f"{now:.1f}% vs {past:.1f}% in the previous 3 years ({diff:+.1f} pp)",
                f"WARN if the share moves more than {DRIFT_SHARE_PP:.0f} pp [A24]",
                WARN if abs(diff) > DRIFT_SHARE_PP else PASS)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--as-of", default=None,
                        help="check date YYYY-MM-DD (default: today); used only for D8 freshness")
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(date.today())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly = fc.with_dates(ev.load_monthly_assault())
    print(f"[monitor] check date {as_of:%Y-%m-%d}; processed panel "
          f"{monthly['date'].min():%Y-%m} to {monthly['date'].max():%Y-%m}")

    log = Log()
    info = data_checks(log, monthly, as_of)
    scored, settled_status = performance_checks(log)
    drift_checks(log, monthly, info["settled_until"])

    checks = log.frame()
    data_fail = (checks["Area"].eq("Data") & checks["Status"].eq(FAIL)).any()
    perf_fail = any(s == FAIL for s in settled_status.values())
    overall = worst(checks["Status"])

    promote = not (data_fail or perf_fail)
    if perf_fail:
        retrain = "Investigate first; retrain if the FAIL repeats at the next release or is explained by lasting drift"
    else:
        retrain = ("Not needed: no performance FAIL on settled months. Next planned refresh "
                   "when the panel can be extended (2026 population estimates, [A1])")

    checks.to_csv(OUT_DIR / "M1_monitoring_checks.csv", index=False)
    scored.to_csv(OUT_DIR / "M2_monitoring_scored.csv", index=False)
    summary = {
        "check_date": f"{as_of:%Y-%m-%d}",
        "crime_data_as_at": f"{info['as_at']:%Y-%m-%d}",
        "settled_until": f"{info['settled_until']:%Y-%m}",
        "provisional_months": [f"{d:%Y-%m}" for d in info["provisional"]],
        "overall_status": overall,
        "counts": {s: int((checks["Status"] == s).sum()) for s in (PASS, WARN, FAIL)},
        "promote_forecasts_to_dashboard": bool(promote),
        "retraining_decision": retrain,
        "warnings": [f"{r.Check_id}: {r.Check} - {r.Value}"
                     for r in checks.itertuples() if r.Status == WARN],
        "failures": [f"{r.Check_id}: {r.Check} - {r.Value}"
                     for r in checks.itertuples() if r.Status == FAIL],
        "environment": {"platform": f"{platform.system()} {platform.machine()}",
                        "python": platform.python_version(),
                        "pandas": pd.__version__},
    }
    (OUT_DIR / "M3_monitoring_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== Monitoring result: {overall} "
          f"({summary['counts'][PASS]} PASS, {summary['counts'][WARN]} WARN, "
          f"{summary['counts'][FAIL]} FAIL) ===")
    print(f"Promote forecasts to the dashboard: {'yes' if promote else 'no'}")
    print(f"Retraining: {retrain}")
    print("\n[monitor] wrote Outputs/M1_monitoring_checks.csv, "
          "Outputs/M2_monitoring_scored.csv, Outputs/M3_monitoring_summary.json")


if __name__ == "__main__":
    main()
