"""
forecast.py - Forecasts, saved forecast models and a first check on new data
(Assessment 4)

Run order:
    python Updated_End_to_End_pipeline.py     # panel, Model A, Outputs/R*
    python evaluation.py                      # backtest, Outputs/E2-E5
    python forecast.py                        # this file, Outputs/F1, E6

What it adds
------------
F1  Forecasts for 1, 3, 6 and 12 months ahead from the last month of the
    processed panel, for every region. They are written to
    Outputs/staging/F1_forecasts.csv as a candidate; monitoring.py publishes
    the candidate to Outputs/F1_forecasts.csv, the file the dashboard reads,
    only if the release checks pass (release_gate.py). Each horizon uses the method with the
    lowest backtest RMSE (Outputs/E4_backtest_metrics.csv) and its empirical
    80% interval (Outputs/E5_prediction_intervals.csv). The fitted models'
    forecasts and the two benchmarks are written alongside for comparison.
F2  One model bundle per horizon in Models/ (forecast_h1.joblib ...), trained
    on every target month up to the forecast origin with the backtested set-up
    (same feature variant, alphas and XGBoost parameters as evaluation.py).
E6  First out-of-sample check. The latest crime file (As At 1 June 2026)
    already holds January to April 2026, which the processed panel stops
    short of because population figures end in 2025 ([A1]). No model has
    seen these months. The saved models are kept frozen and scored on them:
    four targets at h = 1 and two at h = 3, six regions each.

Assumptions (README [A21]-[A23])
--------------------------------
[A22] 2026 rates use each region's 2025 population, the latest published.
      The NT grows slowly, so this shifts a rate by well under 1%.
[A23] All 2026 months are within six months of the As At date, so they are
      provisional and will be revised upward as late reports arrive ([A9]).
      E6 therefore reports bias as well as error: a positive bias (forecast
      above the provisional count) is expected and is not by itself a fault.
      E6 is an early warning check, not a replacement for the backtest.

Every feature used for a forecast was known at its origin. The models are
trained only on target months up to the origin and are not refitted on 2026.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone

import evaluation as ev
import Updated_End_to_End_pipeline as pl

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "Outputs"
STAGING_DIR = OUT_DIR / "staging"      # candidate forecasts, published by monitoring.py
MODEL_DIR = BASE_DIR / "Models"
PLOT_DIR = BASE_DIR / "regression_plots"

FLAG_MAP = {"Yes": 1, "No": 0, "-": 0}   # [A8], as in the pipeline


# ---------------------------------------------------------------------------
# Months beyond the processed panel
# ---------------------------------------------------------------------------

def assault_by_region_month(crime: pd.DataFrame) -> pd.DataFrame:
    """Region x month assault counts from rows of the new-schema crime file,
    using the pipeline's own region crosswalk and flag rules."""
    df = pl.rename_columns(crime.copy(), {"Reporting region": "Reporting Region"})
    df["Reporting Region"] = df["Reporting Region"].replace("Unknown", "Top End")  # [A4]
    df["Region"] = df.apply(pl.remap_region, axis=1)                               # [A5]
    df = df[df["Offence category"] == pl.ASSAULT_CATEGORY].copy()
    for col, out in [("Alcohol involvement", "Alcohol_offences"),
                     ("DV involvement", "DV_offences")]:
        df[out] = df[col].map(FLAG_MAP).fillna(0).astype(int) * df["Number of offences"]
    return (df.groupby(["Year", "Month number", "Region"], as_index=False)
              .agg(Assault_offences=("Number of offences", "sum"),
                   Alcohol_offences=("Alcohol_offences", "sum"),
                   DV_offences=("DV_offences", "sum")))


def check_reconstruction(crime: pd.DataFrame, monthly: pd.DataFrame) -> int:
    """Rebuild a year the panel already has and confirm the counts match it
    exactly, so the 2026 months are built the same way as the panel."""
    year = int(monthly["Year"].max())
    rebuilt = assault_by_region_month(crime[crime["Year"] == year])
    merged = rebuilt.merge(monthly[monthly["Year"] == year],
                           on=["Year", "Month number", "Region"],
                           suffixes=("_rebuilt", "_panel"))
    cols = ["Assault_offences", "Alcohol_offences", "DV_offences"]
    mismatches = sum(int((merged[f"{c}_rebuilt"] != merged[f"{c}_panel"]).sum())
                     for c in cols)
    if len(merged) != len(rebuilt) or mismatches:
        raise ValueError(f"Rebuilding {year} from the raw file does not match the "
                         f"processed panel ({mismatches} differing values).")
    print(f"[forecast] {year} rebuilt from the raw file matches the panel exactly "
          f"({len(merged)} region-months)")
    return year


def load_new_months(monthly: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Assault months after the panel's last month, with rates on the latest
    population ([A22])."""
    crime = pl._read_raw("crime_new")
    as_at = str(crime["As At"].iloc[0]) if "As At" in crime else "unknown"
    check_reconstruction(crime, monthly)

    last = monthly["date"].max() if "date" in monthly else None
    crime_dates = pd.to_datetime(dict(year=crime["Year"],
                                      month=crime["Month number"], day=1))
    new = assault_by_region_month(crime[crime_dates > last])
    if new.empty:
        return new, as_at

    latest_year = int(monthly["Year"].max())
    population = (monthly[monthly["Year"] == latest_year]
                  .groupby("Region")["Total_population"].first())
    new["Total_population"] = new["Region"].map(population)                       # [A22]
    new["Quarter"] = (new["Month number"] - 1) // 3 + 1
    new["Total_PAC"] = np.nan
    new["Assault_rate_100k"] = (new["Assault_offences"] / new["Total_population"]
                                * 100_000).round(1)
    return new, as_at


def with_dates(monthly: pd.DataFrame) -> pd.DataFrame:
    out = monthly.copy()
    out["date"] = pd.to_datetime(dict(year=out["Year"],
                                      month=out["Month number"], day=1))
    return out


def future_placeholders(regions, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Empty rows for months not yet observed, so feature rows can be built
    for them. Their target stays NaN."""
    dates = pd.date_range(start, end, freq="MS")
    rows = [{"Year": d.year, "Quarter": (d.month - 1) // 3 + 1,
             "Month number": d.month, "Region": r} for r in regions for d in dates]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# F2: fit and save one bundle per horizon
# ---------------------------------------------------------------------------

def fit_and_save(base: pd.DataFrame, dummy_cols: list[str], variant: str,
                 models: dict, origin: pd.Timestamp) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fitted = {}
    for h in ev.HORIZONS:
        frame = ev.horizon_frame(base, h)
        train = frame[frame["date"] <= origin]
        features = ev.feature_variants(h, dummy_cols)[variant]
        X = train[features].astype(float).to_numpy()
        y = train["log_assault_rate"].to_numpy(float)
        pipes = {name: ev.make_pipe(clone(m)).fit(X, y) for name, m in models.items()}
        fitted[h] = {"models": pipes, "features": features}

        joblib.dump({
            "horizon_months": h,
            "models": pipes,
            "features": features,
            "feature_variant": variant,
            "target": "log1p(assault rate per 100,000); back-transform with expm1",
            "trained_on_target_months": f"{train['date'].min():%Y-%m} to "
                                        f"{train['date'].max():%Y-%m}",
            "train_rows": len(train),
            "versions": {"python": platform.python_version(),
                         "scikit_learn": sklearn.__version__,
                         "xgboost": xgboost.__version__},
        }, MODEL_DIR / f"forecast_h{h}.joblib", compress=3)
    print(f"[forecast] F2 saved Models/forecast_h{{{','.join(map(str, ev.HORIZONS))}}}.joblib "
          f"(trained on target months up to {origin:%Y-%m})")
    return fitted


def predict_rows(rows: pd.DataFrame, fitted_h: dict) -> dict[str, np.ndarray]:
    X = rows[fitted_h["features"]].astype(float).to_numpy()
    preds = {name: ev.inv_log(pipe.predict(X)) for name, pipe in fitted_h["models"].items()}
    preds[ev.SEASONAL_NAIVE] = rows["naive_seasonal"].to_numpy(float)
    preds[ev.MOVING_AVERAGE] = rows["naive_ma12"].to_numpy(float)
    return preds


# ---------------------------------------------------------------------------
# F1: forecasts from the origin
# ---------------------------------------------------------------------------

def recommended_by_horizon() -> tuple[dict, pd.DataFrame]:
    """Lowest-RMSE entry per horizon and its interval, from evaluation.py."""
    metrics_path = OUT_DIR / "E4_backtest_metrics.csv"
    pi_path = OUT_DIR / "E5_prediction_intervals.csv"
    if not (metrics_path.exists() and pi_path.exists()):
        raise FileNotFoundError("Run evaluation.py first: forecast.py reads "
                                "Outputs/E4_backtest_metrics.csv and "
                                "Outputs/E5_prediction_intervals.csv.")
    metrics = pd.read_csv(metrics_path)
    winners = (metrics.sort_values("RMSE").groupby("Horizon_months")
               .first()["Model"].to_dict())
    intervals = pd.read_csv(pi_path).set_index("Horizon_months")
    return winners, intervals


def make_forecasts(base_ext: pd.DataFrame, fitted: dict, origin: pd.Timestamp,
                   winners: dict, intervals: pd.DataFrame, as_at: str) -> pd.DataFrame:
    records = []
    for h in ev.HORIZONS:
        target = origin + pd.DateOffset(months=h)
        frame = ev.horizon_frame(base_ext, h, require_target=False)
        rows = frame[frame["date"] == target]
        preds = predict_rows(rows, fitted[h])
        pick = winners[h]
        lo_off = float(intervals.loc[h, "Lower_offset"])
        hi_off = float(intervals.loc[h, "Upper_offset"])
        for name, values in preds.items():
            for region, value in zip(rows["Region"], values):
                recommended = name == pick
                records.append({
                    "Origin": f"{origin:%Y-%m}", "Horizon_months": h,
                    "Target_month": f"{target:%Y-%m}", "Region": region,
                    "Model": name, "Recommended": recommended,
                    "Forecast_rate_100k": round(float(value), 1),
                    "Lower_80": round(max(0.0, value + lo_off), 1) if recommended else np.nan,
                    "Upper_80": round(value + hi_off, 1) if recommended else np.nan,
                    "Data_as_at": as_at,
                })
    out = pd.DataFrame(records)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(STAGING_DIR / "F1_forecasts.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# E6: frozen models scored on months none of them has seen
# ---------------------------------------------------------------------------

def holdout_check(base_ext: pd.DataFrame, fitted: dict, origin: pd.Timestamp,
                  last_actual: pd.Timestamp, winners: dict,
                  intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for h in ev.HORIZONS:
        frame = ev.horizon_frame(base_ext, h)        # rows with a known 2026 target
        targets = pd.date_range(origin + pd.DateOffset(months=h), last_actual, freq="MS")
        for target in targets:
            rows = frame[frame["date"] == target]
            if rows.empty:
                continue
            preds = predict_rows(rows, fitted[h])
            for name, values in preds.items():
                for region, actual, value in zip(rows["Region"],
                                                 rows["Assault_rate_100k"], values):
                    records.append({"origin": target - pd.DateOffset(months=h),
                                    "target": target, "h": h, "Model": name,
                                    "Region": region, "actual": float(actual),
                                    "pred": float(value)})
    preds = pd.DataFrame(records)
    if preds.empty:
        return preds, pd.DataFrame()

    backtest = pd.read_csv(OUT_DIR / "E4_backtest_metrics.csv").set_index(
        ["Model", "Horizon_months"])["RMSE"]
    rows = []
    for (model, h), grp in preds.groupby(["Model", "h"]):
        m = ev.rate_metrics(grp["actual"].to_numpy(float), grp["pred"].to_numpy(float))
        residual = grp["actual"] - grp["pred"]
        row = {"Model": model, "Horizon_months": h, **m,
               "Bias_forecast_minus_actual": float(-residual.mean()),
               "Backtest_RMSE": float(backtest.get((model, h), np.nan)),
               "Targets": f"{grp['target'].min():%Y-%m} to {grp['target'].max():%Y-%m}",
               "Recommended": model == winners[h]}
        if model == winners[h]:
            lo = float(intervals.loc[h, "Lower_offset"])
            hi = float(intervals.loc[h, "Upper_offset"])
            inside = (residual >= lo) & (residual <= hi)
            row["Interval_80_coverage_%"] = float(inside.mean() * 100)
        rows.append(row)
    metrics = (pd.DataFrame(rows)
               .sort_values(["Horizon_months", "RMSE"]).reset_index(drop=True))

    out = preds.copy()
    out["origin"] = out["origin"].dt.strftime("%Y-%m")
    out["target"] = out["target"].dt.strftime("%Y-%m")
    out["actual_status"] = "provisional"                          # [A23]
    out.to_csv(OUT_DIR / "E6_holdout_2026_predictions.csv", index=False)
    metrics.to_csv(OUT_DIR / "E6_holdout_2026_metrics.csv", index=False)
    return preds, metrics


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_forecasts(base_ext: pd.DataFrame, forecasts: pd.DataFrame,
                   origin: pd.Timestamp, path: Path) -> None:
    rec = forecasts[forecasts["Recommended"]].copy()
    rec["date"] = pd.to_datetime(rec["Target_month"])
    regions = [r for r in pl.REGION_ORDER if r in set(rec["Region"])]
    colours = {"XGBoost": "#E53935", ev.MOVING_AVERAGE: "#1E88E5",
               ev.SEASONAL_NAIVE: "#546E7A"}
    start = origin - pd.DateOffset(months=35)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for ax, region in zip(axes.ravel(), regions):
        hist = base_ext[(base_ext["Region"] == region) & (base_ext["date"] >= start)]
        settled = hist[hist["date"] <= origin]
        later = hist[(hist["date"] > origin) & hist["Assault_rate_100k"].notna()]
        ax.plot(settled["date"], settled["Assault_rate_100k"], color="#263238",
                linewidth=1.4, label="Observed (to forecast origin)")
        if not later.empty:
            ax.scatter(later["date"], later["Assault_rate_100k"], facecolors="none",
                       edgecolors="#263238", s=28, zorder=3,
                       label="2026 provisional actuals (not used)")
        for model, grp in rec[rec["Region"] == region].groupby("Model"):
            ax.errorbar(grp["date"], grp["Forecast_rate_100k"],
                        yerr=[grp["Forecast_rate_100k"] - grp["Lower_80"],
                              grp["Upper_80"] - grp["Forecast_rate_100k"]],
                        fmt="o", color=colours.get(model, "#8E24AA"), capsize=3,
                        label=f"{model} forecast, 80% interval")
        ax.axvline(origin, color="#9E9E9E", linestyle="--", linewidth=1)
        ax.set_title(region, fontsize=11)
        ax.grid(alpha=0.3)
        # One label every six months, so the shared date axis stays readable
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.tick_params(axis="x", labelsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Assaults per 100,000")
    handles, labels = {}, {}
    for ax in axes.ravel():
        for hnd, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, hnd)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3,
               fontsize=9, frameon=False)
    fig.suptitle(f"Forecasts from {origin:%B %Y} at 1, 3, 6 and 12 months "
                 f"(recommended method per horizon)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly = with_dates(ev.load_monthly_assault())
    origin = monthly["date"].max()
    base = ev.build_base_panel(monthly.drop(columns="date"))
    dummy_cols = [c for c in base.columns if c.startswith("Reg_")]
    print(f"[forecast] forecast origin = last month of the processed panel: {origin:%Y-%m}")

    new, as_at = load_new_months(monthly)
    last_actual = (with_dates(new)["date"].max() if not new.empty else origin)
    print(f"[forecast] crime file As At {as_at}: {len(new)} region-months after the "
          f"panel ({'none' if new.empty else f'{origin + pd.DateOffset(months=1):%Y-%m} to {last_actual:%Y-%m}'})")

    horizon_end = origin + pd.DateOffset(months=max(ev.HORIZONS))
    filler = future_placeholders(sorted(monthly["Region"].unique()),
                                 last_actual + pd.DateOffset(months=1), horizon_end)
    extended = pd.concat([monthly.drop(columns="date"), new, filler], ignore_index=True)
    base_ext = ev.build_base_panel(extended)

    variant, _, ridge_alpha, lasso_alpha = ev.choose_setup(base, dummy_cols)
    models = ev.build_models(ridge_alpha, lasso_alpha)
    print(f"[forecast] set-up from evaluation.py: {variant}, Ridge {ridge_alpha}, "
          f"Lasso {lasso_alpha}")

    fitted = fit_and_save(base, dummy_cols, variant, models, origin)
    winners, intervals = recommended_by_horizon()

    forecasts = make_forecasts(base_ext, fitted, origin, winners, intervals, as_at)
    print("\n=== F1 recommended forecasts (assault rate per 100k, 80% interval) ===")
    show = forecasts[forecasts["Recommended"]].pivot_table(
        index="Region", columns="Target_month", values="Forecast_rate_100k")
    print(show.round(1).to_string())
    print("   methods: " + ", ".join(f"h={h}: {m}" for h, m in winners.items()))

    _, metrics = holdout_check(base_ext, fitted, origin, last_actual, winners, intervals)
    if not metrics.empty:
        print("\n=== E6 frozen models on 2026 months (provisional actuals, [A23]) ===")
        print(metrics[["Model", "Horizon_months", "n", "RMSE", "MAE",
                       "Bias_forecast_minus_actual", "Backtest_RMSE",
                       "Interval_80_coverage_%"]]
              .to_string(index=False, float_format="%.1f"))

    plot_forecasts(base_ext, forecasts, origin, PLOT_DIR / "R9_forecasts_2026.png")

    summary = {
        "forecast_origin": f"{origin:%Y-%m}",
        "crime_data_as_at": as_at,
        "new_months_scored": sorted(with_dates(new)["date"].dt.strftime("%Y-%m").unique().tolist())
                             if not new.empty else [],
        "feature_variant": variant,
        "ridge_alpha": ridge_alpha, "lasso_alpha": lasso_alpha,
        "recommended_by_horizon": {str(h): m for h, m in winners.items()},
        "model_files": [f"Models/forecast_h{h}.joblib" for h in ev.HORIZONS],
        "environment": {"platform": f"{platform.system()} {platform.machine()}",
                        "python": platform.python_version(),
                        "pandas": pd.__version__, "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__,
                        "xgboost": xgboost.__version__},
    }
    (OUT_DIR / "forecast_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n[forecast] wrote the candidate Outputs/staging/F1_forecasts.csv, "
          "Outputs/E6_holdout_2026_*.csv, Outputs/forecast_run_summary.json, "
          "regression_plots/R9_forecasts_2026.png")
    print("[forecast] run monitoring.py next: it publishes the candidate to "
          "Outputs/F1_forecasts.csv only if the release checks pass")


if __name__ == "__main__":
    main()
