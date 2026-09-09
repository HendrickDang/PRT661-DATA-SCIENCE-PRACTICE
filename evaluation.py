"""
evaluation.py - Forecast evaluation module

Purpose
-------
Addresses the Assessment 1 feedback on the modelling plan by supplying the
pieces the main pipeline does not yet have:

  E1  Naive baselines (seasonal naive, 12-month moving average) so that model
      skill is measured against a benchmark rather than against other models.
  E2  Leakage-free feature-variant selection - the variant is chosen on
      cross-validation folds, never on the held-out test set.
  E3  Scaling inside a Pipeline so each CV fold is scaled on its own training
      rows only.
  E4  Rolling-origin backtesting at forecast horizons h = 1, 3, 6 and 12
      months, which is what a 3-12 month operational forecast requires.
  E5  Empirical prediction intervals from backtest residual quantiles.
  E6  Explicit structural-break indicators (SerPro Dec-2023, ANZSOC Apr-2025).
  E7  Metrics reported on the original assault-rate scale, not the log scale.
  E8  Provisional months excluded from evaluation, as the project plan states.

Run:  python evaluation.py
Reads dataset/processed/nt_crime_merged_2015_2025.csv, writes metrics CSVs to
outputs/ and figures to regression_plots/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor  # noqa: F401  (documented alternative)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
PROCESSED = BASE_DIR / "dataset" / "processed" / "nt_crime_merged_2015_2025.csv"
OUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = BASE_DIR / "regression_plots"

SERPRO_BREAK = pd.Timestamp("2023-12-01")   # PROMIS -> SerPro
ANZSOC_BREAK = pd.Timestamp("2025-04-01")   # ANZSOC reclassification
PROVISIONAL_MONTHS = 6                      # [A9] - excluded from evaluation

LAGS = (1, 3, 12)
HORIZONS = (1, 3, 6, 12)


# ---------------------------------------------------------------------------
# Panel construction (standalone - does not import the main pipeline)
# ---------------------------------------------------------------------------

def load_monthly_assault(path: Path = PROCESSED) -> pd.DataFrame:
    """Region x month assault aggregate, mirroring the main pipeline."""
    panel = pd.read_csv(path)
    assault = panel[panel["Offence category"] == "02 Assault"]

    monthly = (assault.groupby(["Year", "Quarter", "Month number", "Region"])
               .agg(Assault_offences=("Number of offences", "sum"),
                    Alcohol_offences=("Alcohol_offences", "sum"),
                    DV_offences=("DV_offences", "sum"),
                    Total_PAC=("Total PAC", "first"),
                    Total_population=("Total_population", "first"))
               .reset_index())

    monthly["Assault_rate_100k"] = (monthly["Assault_offences"]
                                    / monthly["Total_population"] * 100_000)
    monthly["alcohol_per_capita"] = (monthly["Total_PAC"]
                                     / monthly["Total_population"])
    return monthly


def build_panel(monthly: pd.DataFrame) -> pd.DataFrame:
    """Complete region x month grid with lags, seasonality and break flags."""
    obs = monthly.copy()
    obs["date"] = pd.to_datetime(dict(year=obs["Year"],
                                      month=obs["Month number"], day=1))

    grid = pd.MultiIndex.from_product(
        [sorted(obs["Region"].unique()),
         pd.date_range(obs["date"].min(), obs["date"].max(), freq="MS")],
        names=["Region", "date"],
    ).to_frame(index=False)

    p = grid.merge(obs.drop(columns=["Year", "Quarter", "Month number"]),
                   on=["Region", "date"], how="left")
    p["Year"] = p["date"].dt.year
    p["Month number"] = p["date"].dt.month
    p = p.sort_values(["Region", "date"]).reset_index(drop=True)

    # E7: keep the rate on its natural scale; log only as the model target
    p["log_assault_rate"] = np.log1p(p["Assault_rate_100k"])

    p["sin_month"] = np.sin(2 * np.pi * p["Month number"] / 12)
    p["cos_month"] = np.cos(2 * np.pi * p["Month number"] / 12)
    p["Season"] = p["Month number"].isin([11, 12, 1, 2, 3, 4]).astype(int)

    # E6: structural breaks as explicit regressors
    p["post_serpro"] = (p["date"] >= SERPRO_BREAK).astype(int)
    p["post_anzsoc"] = (p["date"] >= ANZSOC_BREAK).astype(int)

    for lag in LAGS:
        p[f"assault_rate_lag{lag}"] = p.groupby("Region")["Assault_rate_100k"].shift(lag)

    # seasonal naive reference: same month, previous year
    p["naive_seasonal"] = p.groupby("Region")["Assault_rate_100k"].shift(12)
    # 12-month moving average of the 12 months ending one month before t
    p["naive_ma12"] = (p.groupby("Region")["Assault_rate_100k"]
                       .transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean()))

    # E8: flag the trailing provisional window
    last_obs = p.loc[p["Assault_offences"].notna(), "date"].max()
    p["is_provisional"] = p["date"] > (last_obs - pd.DateOffset(months=PROVISIONAL_MONTHS))

    dummies = pd.get_dummies(p["Region"], prefix="Reg").drop(
        columns=["Reg_Greater Darwin"], errors="ignore")
    p = pd.concat([p, dummies.astype(int)], axis=1)

    required = ["log_assault_rate", "naive_seasonal", "naive_ma12"] + \
               [f"assault_rate_lag{l}" for l in LAGS]
    p = p.dropna(subset=required).reset_index(drop=True)
    return p


# ---------------------------------------------------------------------------
# Metrics - reported on the assault-rate scale (E7)
# ---------------------------------------------------------------------------

def rate_metrics(y_true_rate: np.ndarray, y_pred_rate: np.ndarray) -> dict:
    mask = np.isfinite(y_true_rate) & np.isfinite(y_pred_rate)
    yt, yp = y_true_rate[mask], y_pred_rate[mask]
    nonzero = yt > 0
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAPE_%": float(np.mean(np.abs((yt[nonzero] - yp[nonzero])
                                       / yt[nonzero])) * 100),
        "n": int(mask.sum()),
    }


def inv_log(v: np.ndarray) -> np.ndarray:
    """Back-transform log1p predictions to the rate scale."""
    return np.expm1(v)


# ---------------------------------------------------------------------------
# E1: baselines
# ---------------------------------------------------------------------------

def baseline_scores(frame: pd.DataFrame) -> pd.DataFrame:
    truth = frame["Assault_rate_100k"].to_numpy(float)
    rows = [
        {"Model": "Seasonal naive (t-12)", **rate_metrics(truth,
            frame["naive_seasonal"].to_numpy(float))},
        {"Model": "Moving average (12m)", **rate_metrics(truth,
            frame["naive_ma12"].to_numpy(float))},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E2 + E3: leakage-free variant selection with scaling inside the Pipeline
# ---------------------------------------------------------------------------

def make_pipe(model):
    """E3 - scaler fitted per fold, never across folds."""
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def cv_rmse(model, cv_frame: pd.DataFrame, features: list[str],
            n_splits: int = 5) -> float:
    """TimeSeriesSplit CV RMSE on the rate scale, scaled inside each fold."""
    frame = cv_frame.sort_values(["date", "Region"]).reset_index(drop=True)
    X = frame[features].astype(float).to_numpy()
    y_log = frame["log_assault_rate"].to_numpy(float)
    y_rate = frame["Assault_rate_100k"].to_numpy(float)

    errs = []
    for tr_idx, te_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        pipe = make_pipe(model)
        pipe.fit(X[tr_idx], y_log[tr_idx])
        pred_rate = inv_log(pipe.predict(X[te_idx]))
        errs.append(np.sqrt(mean_squared_error(y_rate[te_idx], pred_rate)))
    return float(np.mean(errs))


def select_variant(cv_frame: pd.DataFrame, variants: dict[str, list[str]]) -> tuple:
    """E2 - choose the feature set on CV folds only. The test set is untouched."""
    scores = {name: cv_rmse(LinearRegression(), cv_frame, feats)
              for name, feats in variants.items()}
    best = min(scores, key=scores.get)
    table = (pd.DataFrame({"Variant": list(scores), "CV_RMSE": list(scores.values())})
             .sort_values("CV_RMSE").reset_index(drop=True))
    return best, variants[best], table


# ---------------------------------------------------------------------------
# E4: rolling-origin backtest at multiple horizons
# ---------------------------------------------------------------------------

def rolling_origin_backtest(panel: pd.DataFrame, features: list[str],
                            models: dict, horizons=HORIZONS,
                            min_train_months: int = 60,
                            step: int = 3) -> pd.DataFrame:
    """
    Expanding-window backtest. At each origin the model is refit on every month
    strictly before the origin, then asked to predict the month h steps ahead.
    Features at the target month use only lagged information, so no future
    value enters the prediction.
    """
    dates = np.array(sorted(panel["date"].unique()))
    eval_pool = panel[~panel["is_provisional"]]
    valid_dates = np.array(sorted(eval_pool["date"].unique()))

    records = []
    start = min_train_months
    for i in range(start, len(dates), step):
        origin = dates[i]
        train = panel[panel["date"] < origin]
        if len(train) < min_train_months:
            continue

        X_tr = train[features].astype(float).to_numpy()
        y_tr = train["log_assault_rate"].to_numpy(float)

        fitted = {}
        for name, model in models.items():
            pipe = make_pipe(model)
            pipe.fit(X_tr, y_tr)
            fitted[name] = pipe

        for h in horizons:
            tgt_idx = i + h - 1
            if tgt_idx >= len(dates):
                continue
            target_date = dates[tgt_idx]
            if target_date not in valid_dates:      # E8: skip provisional
                continue
            tgt = panel[panel["date"] == target_date]
            if tgt.empty:
                continue

            X_te = tgt[features].astype(float).to_numpy()
            truth = tgt["Assault_rate_100k"].to_numpy(float)

            for name, pipe in fitted.items():
                pred = inv_log(pipe.predict(X_te))
                for r, t, p in zip(tgt["Region"], truth, pred):
                    records.append({"origin": origin, "target": target_date,
                                    "h": h, "Model": name, "Region": r,
                                    "actual": t, "pred": p})

            for name, col in (("Seasonal naive (t-12)", "naive_seasonal"),
                              ("Moving average (12m)", "naive_ma12")):
                pred = tgt[col].to_numpy(float)
                for r, t, p in zip(tgt["Region"], truth, pred):
                    records.append({"origin": origin, "target": target_date,
                                    "h": h, "Model": name, "Region": r,
                                    "actual": t, "pred": p})

    return pd.DataFrame(records)


def summarise_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, h), grp in bt.groupby(["Model", "h"]):
        m = rate_metrics(grp["actual"].to_numpy(float), grp["pred"].to_numpy(float))
        rows.append({"Model": model, "Horizon_months": h, **m})
    out = pd.DataFrame(rows).sort_values(["Horizon_months", "RMSE"])
    return out.reset_index(drop=True)


def skill_vs_baseline(summary: pd.DataFrame,
                      baseline: str = "Seasonal naive (t-12)") -> pd.DataFrame:
    """Percentage RMSE improvement over the baseline. Negative = worse."""
    base = summary[summary["Model"] == baseline].set_index("Horizon_months")["RMSE"]
    out = summary.copy()
    out["Baseline_RMSE"] = out["Horizon_months"].map(base)
    out["Skill_%"] = (1 - out["RMSE"] / out["Baseline_RMSE"]) * 100
    return out


# ---------------------------------------------------------------------------
# E5: empirical prediction intervals from backtest residuals
# ---------------------------------------------------------------------------

def prediction_intervals(bt: pd.DataFrame, model: str,
                         lower_q: float = 0.10,
                         upper_q: float = 0.90) -> pd.DataFrame:
    """
    Interval width per horizon, taken from the distribution of backtest
    residuals. Distribution-free, so it makes no normality assumption.
    """
    sub = bt[bt["Model"] == model].copy()
    sub["residual"] = sub["actual"] - sub["pred"]
    rows = []
    for h, grp in sub.groupby("h"):
        lo, hi = grp["residual"].quantile([lower_q, upper_q])
        cover = ((grp["actual"] >= grp["pred"] + lo)
                 & (grp["actual"] <= grp["pred"] + hi)).mean() * 100
        rows.append({"Horizon_months": h,
                     "Lower_offset": float(lo), "Upper_offset": float(hi),
                     "Interval_width": float(hi - lo),
                     "Empirical_coverage_%": float(cover)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    monthly = load_monthly_assault()
    panel = build_panel(monthly)
    print(f"[eval] panel: {len(panel)} region-month rows, "
          f"{panel['date'].min():%Y-%m} to {panel['date'].max():%Y-%m}")
    print(f"[eval] provisional rows excluded from evaluation: "
          f"{int(panel['is_provisional'].sum())}")

    dummy_cols = [c for c in panel.columns if c.startswith("Reg_")]
    base = ["sin_month", "cos_month", "Season",
            "assault_rate_lag1", "assault_rate_lag3", "assault_rate_lag12"]
    breaks = ["post_serpro", "post_anzsoc"]

    variants = {
        "V1: Temporal": base,
        "V2: + Region": base + dummy_cols,
        "V3: + Crime ctx": base + ["Alcohol_offences", "DV_offences"],
        "V4: Full (no PAC)": base + ["Alcohol_offences", "DV_offences"] + dummy_cols,
        "V5: V2 + breaks": base + dummy_cols + breaks,
    }

    cv_frame = panel[panel["Year"] <= 2022]
    best_name, features, variant_table = select_variant(cv_frame, variants)
    print(f"\n[eval] E2 variant selected on CV only -> {best_name}")
    print(variant_table.to_string(index=False))
    variant_table.to_csv(OUT_DIR / "E2_variant_selection_cv.csv", index=False)

    models = {
        "Linear": LinearRegression(),
        "Ridge (a=1.0)": Ridge(alpha=1.0),
        "Lasso (a=0.01)": Lasso(alpha=0.01, max_iter=5000),
    }

    print("\n[eval] E4 running rolling-origin backtest ...")
    bt = rolling_origin_backtest(panel, features, models)
    bt.to_csv(OUT_DIR / "E4_backtest_predictions.csv", index=False)

    summary = summarise_backtest(bt)
    summary = skill_vs_baseline(summary)
    summary.to_csv(OUT_DIR / "E4_backtest_metrics.csv", index=False)

    print("\n=== Forecast accuracy by horizon (assault rate per 100k) ===")
    print(summary[["Model", "Horizon_months", "MAE", "RMSE",
                   "MAPE_%", "Skill_%", "n"]].to_string(index=False,
                                                        float_format="%.3f"))

    best_model = (summary[summary["Horizon_months"] == 3]
                  .sort_values("RMSE").iloc[0]["Model"])
    pi = prediction_intervals(bt, best_model)
    pi.to_csv(OUT_DIR / "E5_prediction_intervals.csv", index=False)
    print(f"\n=== E5 prediction intervals for {best_model} (10th-90th pct) ===")
    print(pi.to_string(index=False, float_format="%.3f"))

    print(f"\n[eval] outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
