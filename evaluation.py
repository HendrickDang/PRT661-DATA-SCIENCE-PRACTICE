"""
evaluation.py - Forecast evaluation module

Purpose
-------
Addresses the Assessment 1 feedback on the modelling plan by supplying the
pieces the main pipeline does not have:

  E1  Naive baselines (seasonal naive, 12-month moving average) so that model
      skill is measured against a benchmark rather than against other models.
  E2  Leakage-free feature-variant selection - the variant is chosen on
      cross-validation folds, never on the held-out test set.
  E3  Scaling inside a Pipeline so each CV fold is scaled on its own training
      rows only.
  E4  Rolling-origin backtesting at forecast horizons h = 1, 3, 6 and 12
      months, which is what a 3-12 month operational forecast requires.
  E5  Empirical prediction intervals from backtest residual quantiles, with
      coverage measured out of sample.
  E6  Explicit structural-break indicators (SerPro Dec-2023, ANZSOC Apr-2025).
  E7  Metrics reported on the original assault-rate scale, not the log scale.
  E8  Provisional months excluded from evaluation, as the project plan states.
  E9  Every feature and every baseline is known at the forecast origin. A
      forecast made h months ahead uses nothing more recent than h months
      before the target. See "What changed in Assessment 3" below.
  E10 Paired bootstrap on the difference between the best model and the
      12-month moving average, so a small gap is not reported as a win.

What changed in Assessment 3
----------------------------
The Assessment 2 version refitted the models at each origin but built the
features at the TARGET month. A "3-month-ahead" forecast therefore still used
the assault rate from one month before the target, and the moving average used
the twelve months ending one month before the target. Neither is available
three months out. In addition h = 1 was scored on Jan/Apr/Jul/Oct targets and
h = 3, 6, 12 on Mar/Jun/Sep/Dec targets, so the horizons were not compared on
the same months.

This version forecasts directly: for horizon h the lag features are the rate
h and h+2 months before the target plus the rate 12 months before, the moving
average ends h months before the target, and one model is fitted per horizon.
Every month from the first origin onward is scored, and the models are refitted
every 3 months on an expanding window of at least 60 months.

Run:  python evaluation.py
Reads dataset/processed/nt_crime_merged_2015_2025.csv, writes result tables to
Outputs/ and one figure to regression_plots/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
PROCESSED = BASE_DIR / "dataset" / "processed" / "nt_crime_merged_2015_2025.csv"
OUT_DIR = BASE_DIR / "Outputs"      # capital O: the folder committed to the repo
PLOT_DIR = BASE_DIR / "regression_plots"

SERPRO_BREAK = pd.Timestamp("2023-12-01")   # PROMIS -> SerPro
ANZSOC_BREAK = pd.Timestamp("2025-04-01")   # ANZSOC reclassification
PROVISIONAL_MONTHS = 6                      # [A9] - excluded from evaluation

HORIZONS = (1, 3, 6, 12)
MIN_TRAIN_MONTHS = 60       # shortest training window, at every horizon
REFIT_EVERY = 3             # months between refits (expanding window)
TRAIN_END_YEAR = 2022       # CV for variant and alpha selection stops here
ALPHAS = (0.01, 0.1, 1, 10, 100)

SEASONAL_NAIVE = "Seasonal naive (t-12)"
MOVING_AVERAGE = "Moving average (12m)"


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

    # Rounded to one decimal, as the main pipeline does, so both scripts model
    # exactly the same target.
    monthly["Assault_rate_100k"] = (monthly["Assault_offences"]
                                    / monthly["Total_population"] * 100_000).round(1)
    return monthly


def build_base_panel(monthly: pd.DataFrame) -> pd.DataFrame:
    """Complete region x month grid with the columns that do not depend on the
    forecast horizon: target, seasonality, break flags and region dummies."""
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

    # E8: flag the trailing provisional window
    last_obs = p.loc[p["Assault_offences"].notna(), "date"].max()
    p["is_provisional"] = p["date"] > (last_obs - pd.DateOffset(months=PROVISIONAL_MONTHS))

    dummies = pd.get_dummies(p["Region"], prefix="Reg").drop(
        columns=["Reg_Greater Darwin"], errors="ignore")
    return pd.concat([p, dummies.astype(int)], axis=1)


def horizon_lags(h: int) -> list[int]:
    """Lags of the assault rate usable h months ahead.

    The most recent value known at the origin (h months back), the value two
    months before that, and the same month last year. At h = 1 this is the
    pipeline's lag 1 / 3 / 12 set. At h = 12 the first and last coincide.
    """
    if h > 12:
        raise ValueError("The seasonal-naive benchmark needs h <= 12.")
    return sorted({h, h + 2, 12})


def horizon_frame(base: pd.DataFrame, h: int,
                  require_target: bool = True) -> pd.DataFrame:
    """E9 - rows for a direct h-month-ahead forecast. Each row is a target
    month; every feature and baseline on it was observed at least h months
    earlier. forecast.py sets require_target=False to build rows for future
    months whose assault rate is not known yet."""
    f = base.copy()
    by_region = f.groupby("Region")

    for lag in horizon_lags(h):
        f[f"assault_rate_lag{lag}"] = by_region["Assault_rate_100k"].shift(lag)

    # Crime context, lagged to the origin. The same-month counts are part of
    # the target (pipeline [A18]), so they never enter unlagged.
    for col in ("Alcohol_offences", "DV_offences"):
        f[f"{col}_lag{h}"] = by_region[col].shift(h)

    # E1 baselines, both known at the origin:
    f["naive_seasonal"] = by_region["Assault_rate_100k"].shift(12)
    f["naive_ma12"] = by_region["Assault_rate_100k"].transform(
        lambda s: s.shift(h).rolling(12, min_periods=6).mean())

    required = ((["log_assault_rate"] if require_target else [])
                + ["naive_seasonal", "naive_ma12"]
                + [f"assault_rate_lag{lag}" for lag in horizon_lags(h)]
                + [f"Alcohol_offences_lag{h}", f"DV_offences_lag{h}"])
    return f.dropna(subset=required).reset_index(drop=True)


def feature_variants(h: int, dummy_cols: list[str]) -> dict[str, list[str]]:
    base = (["sin_month", "cos_month", "Season"]
            + [f"assault_rate_lag{lag}" for lag in horizon_lags(h)])
    crime = [f"Alcohol_offences_lag{h}", f"DV_offences_lag{h}"]
    breaks = ["post_serpro", "post_anzsoc"]
    return {
        "V1: Temporal": base,
        "V2: + Region": base + dummy_cols,
        "V3: + Crime ctx": base + crime,
        "V4: Full (no PAC)": base + crime + dummy_cols,
        "V5: V2 + breaks": base + dummy_cols + breaks,
    }


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
# E2 + E3: leakage-free selection with scaling inside the Pipeline
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
        pipe = make_pipe(clone(model))
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
             .sort_values("CV_RMSE", kind="stable").reset_index(drop=True))
    table["Selected"] = table["Variant"] == best
    return best, table


def tune_alpha(model_cls, cv_frame: pd.DataFrame, features: list[str],
               **kwargs) -> float:
    """Regularisation strength chosen on the same training-period CV."""
    scores = {a: cv_rmse(model_cls(alpha=a, **kwargs), cv_frame, features)
              for a in ALPHAS}
    return min(scores, key=scores.get)


def choose_setup(base: pd.DataFrame, dummy_cols: list[str]) -> tuple:
    """E2 - feature variant and Ridge/Lasso alphas, all chosen on h = 1 CV over
    the training period (to 2022). Shared with forecast.py so the saved
    forecast models use exactly the backtested set-up."""
    cv_frame = horizon_frame(base, 1)
    cv_frame = cv_frame[cv_frame["Year"] <= TRAIN_END_YEAR]
    variants = feature_variants(1, dummy_cols)
    best_name, variant_table = select_variant(cv_frame, variants)
    ridge_alpha = tune_alpha(Ridge, cv_frame, variants[best_name])
    lasso_alpha = tune_alpha(Lasso, cv_frame, variants[best_name], max_iter=10000)
    return best_name, variant_table, ridge_alpha, lasso_alpha


def build_models(ridge_alpha: float, lasso_alpha: float) -> dict:
    return {
        "Linear": LinearRegression(),
        f"Ridge (a={ridge_alpha})": Ridge(alpha=ridge_alpha),
        f"Lasso (a={lasso_alpha})": Lasso(alpha=lasso_alpha, max_iter=10000),
        # Parameters match the grid point the pipeline selects. The scaler in
        # make_pipe is inert for a tree model.
        "XGBoost": XGBRegressor(
            n_estimators=800,
            max_depth=3,
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=3.0,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            objective="reg:squarederror",
            random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# E4 + E9: rolling-origin backtest with genuine horizons
# ---------------------------------------------------------------------------

def first_origin(frames: dict[int, pd.DataFrame],
                 min_train_months: int = MIN_TRAIN_MONTHS) -> pd.Timestamp:
    """Earliest origin at which EVERY horizon has min_train_months of training
    rows, so all horizons share one set of origins."""
    return max(sorted(f["date"].unique())[min_train_months - 1]
               for f in frames.values())


def rolling_origin_backtest(base: pd.DataFrame, variant: str, models: dict,
                            dummy_cols: list[str], horizons=HORIZONS,
                            min_train_months: int = MIN_TRAIN_MONTHS,
                            refit_every: int = REFIT_EVERY) -> pd.DataFrame:
    """
    Expanding-window backtest. An origin is the last month whose data is in
    hand. At each origin, and for each horizon h, the forecast is for the month
    h steps later, from a model trained only on target months up to the origin
    and from features observed at or before the origin. Models are refitted
    every `refit_every` origins and reused in between.
    """
    frames = {h: horizon_frame(base, h) for h in horizons}
    start = first_origin(frames, min_train_months)
    last_month = base["date"].max()
    origins = pd.date_range(start, last_month, freq="MS")

    records = []
    fitted: dict[int, dict] = {}
    for step, origin in enumerate(origins):
        refit = step % refit_every == 0
        for h in horizons:
            frame = frames[h]
            features = feature_variants(h, dummy_cols)[variant]

            if refit:
                train = frame[frame["date"] <= origin]
                X_tr = train[features].astype(float).to_numpy()
                y_tr = train["log_assault_rate"].to_numpy(float)
                # clone: one independent model per horizon
                fitted[h] = {name: make_pipe(clone(model)).fit(X_tr, y_tr)
                             for name, model in models.items()}

            target_date = origin + pd.DateOffset(months=h)
            tgt = frame[(frame["date"] == target_date)
                        & ~frame["is_provisional"]]          # E8
            if tgt.empty:
                continue

            X_te = tgt[features].astype(float).to_numpy()
            truth = tgt["Assault_rate_100k"].to_numpy(float)

            preds = {name: inv_log(pipe.predict(X_te))
                     for name, pipe in fitted[h].items()}
            preds[SEASONAL_NAIVE] = tgt["naive_seasonal"].to_numpy(float)
            preds[MOVING_AVERAGE] = tgt["naive_ma12"].to_numpy(float)

            for name, pred in preds.items():
                for region, actual, value in zip(tgt["Region"], truth, pred):
                    records.append({"origin": origin, "target": target_date,
                                    "h": h, "Model": name, "Region": region,
                                    "actual": actual, "pred": value})

    return pd.DataFrame(records)


def summarise_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, h), grp in bt.groupby(["Model", "h"]):
        m = rate_metrics(grp["actual"].to_numpy(float), grp["pred"].to_numpy(float))
        rows.append({"Model": model, "Horizon_months": h, **m,
                     "First_target": grp["target"].min().strftime("%Y-%m"),
                     "Last_target": grp["target"].max().strftime("%Y-%m")})
    out = pd.DataFrame(rows).sort_values(["Horizon_months", "RMSE"])
    return out.reset_index(drop=True)


def skill_vs_baseline(summary: pd.DataFrame,
                      baseline: str = SEASONAL_NAIVE) -> pd.DataFrame:
    """Percentage RMSE improvement over the baseline. Negative = worse."""
    base = summary[summary["Model"] == baseline].set_index("Horizon_months")["RMSE"]
    out = summary.copy()
    out["Baseline_RMSE"] = out["Horizon_months"].map(base)
    out["Skill_%"] = (1 - out["RMSE"] / out["Baseline_RMSE"]) * 100
    return out


def common_window(bt: pd.DataFrame) -> pd.DataFrame:
    """Robustness check: keep only target months scored at EVERY horizon, so a
    difference between horizons cannot come from a different set of months."""
    per_h = [set(g["target"]) for _, g in bt.groupby("h")]
    shared = set.intersection(*per_h)
    return bt[bt["target"].isin(shared)]


# ---------------------------------------------------------------------------
# E10: is the best model's lead over the moving average more than noise?
# ---------------------------------------------------------------------------

def paired_bootstrap(bt: pd.DataFrame, challenger: str, reference: str,
                     n_boot: int = 5000, seed: int = 42) -> dict:
    """Bootstrap the RMSE difference (challenger - reference), resampling whole
    target months so the six regions of a month stay together."""
    wide = (bt[bt["Model"].isin([challenger, reference])]
            .pivot_table(index=["target", "Region"], columns="Model", values="pred")
            .join(bt.drop_duplicates(["target", "Region"])
                  .set_index(["target", "Region"])["actual"])
            .dropna().reset_index())
    # Squared errors summed per target month; a resample of months is then a
    # resample of rows of this small table.
    wide["se_challenger"] = (wide["actual"] - wide[challenger]) ** 2
    wide["se_reference"] = (wide["actual"] - wide[reference]) ** 2
    monthly = wide.groupby("target").agg(se_challenger=("se_challenger", "sum"),
                                         se_reference=("se_reference", "sum"),
                                         n=("actual", "size"))
    se_c = monthly["se_challenger"].to_numpy()
    se_r = monthly["se_reference"].to_numpy()
    count = monthly["n"].to_numpy()
    months = monthly.index

    def rmse_gap(idx: np.ndarray) -> float:
        n = count[idx].sum()
        return float(np.sqrt(se_c[idx].sum() / n) - np.sqrt(se_r[idx].sum() / n))

    rng = np.random.default_rng(seed)
    everything = np.arange(len(months))
    gaps = np.array([rmse_gap(rng.integers(0, len(months), len(months)))
                     for _ in range(n_boot)])

    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {"RMSE_difference": rmse_gap(everything),
            "CI95_low": float(lo), "CI95_high": float(hi),
            "Share_of_resamples_challenger_better_%": float((gaps < 0).mean() * 100),
            "Target_months": int(len(months))}


# ---------------------------------------------------------------------------
# E5: empirical prediction intervals, coverage measured out of sample
# ---------------------------------------------------------------------------

def prediction_intervals(bt: pd.DataFrame, model_by_horizon: dict[int, str],
                         lower_q: float = 0.10, upper_q: float = 0.90,
                         min_history_months: int = 12) -> pd.DataFrame:
    """
    80% interval per horizon from the distribution of backtest residuals
    (distribution-free, so no normality assumption).

    The offsets reported use every residual. Coverage is measured out of
    sample: each target month is checked against an interval built only from
    residuals of EARLIER target months. Coverage computed on the same residuals
    that set the interval is close to 80% by construction and says nothing.
    """
    rows = []
    for h, model in model_by_horizon.items():
        sub = bt[(bt["Model"] == model) & (bt["h"] == h)].copy()
        sub["residual"] = sub["actual"] - sub["pred"]
        lo, hi = sub["residual"].quantile([lower_q, upper_q])

        hits, checked = 0, 0
        months = sorted(sub["target"].unique())
        for i, month in enumerate(months):
            if i < min_history_months:
                continue
            past = sub.loc[sub["target"] < month, "residual"]
            p_lo, p_hi = past.quantile([lower_q, upper_q])
            now = sub[sub["target"] == month]
            hits += int(((now["residual"] >= p_lo) & (now["residual"] <= p_hi)).sum())
            checked += len(now)

        rows.append({"Horizon_months": h, "Model": model,
                     "Lower_offset": float(lo), "Upper_offset": float(hi),
                     "Interval_width": float(hi - lo),
                     "Out_of_sample_coverage_%": (hits / checked * 100) if checked else np.nan,
                     "Forecasts_checked": checked})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_rmse_by_horizon(summary: pd.DataFrame, path: Path) -> None:
    colours = {"XGBoost": "#E53935", MOVING_AVERAGE: "#1E88E5",
               SEASONAL_NAIVE: "#78909C", "Linear": "#8E24AA"}
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for model, grp in summary.groupby("Model"):
        grp = grp.sort_values("Horizon_months")
        prefix = model.split(" ")[0]
        colour = colours.get(model, {"Ridge": "#FB8C00", "Lasso": "#43A047"}.get(prefix))
        heavy = model in ("XGBoost", MOVING_AVERAGE)
        ax.plot(grp["Horizon_months"], grp["RMSE"], marker="o", label=model,
                color=colour, linewidth=2.6 if heavy else 1.4)
    ax.set_xticks(list(HORIZONS))
    ax.set(xlabel="Forecast horizon (months ahead)",
           ylabel="RMSE (assault rate per 100,000)",
           title="Rolling-Origin Backtest: RMSE by Horizon")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly = load_monthly_assault()
    base = build_base_panel(monthly)
    dummy_cols = [c for c in base.columns if c.startswith("Reg_")]
    usable = horizon_frame(base, 1)
    print(f"[eval] panel at h=1: {len(usable)} region-month rows, "
          f"{usable['date'].min():%Y-%m} to {usable['date'].max():%Y-%m}")
    print(f"[eval] provisional rows excluded from evaluation: "
          f"{int(usable['is_provisional'].sum())}")

    # E2: variant and alphas chosen on training-period CV at h = 1
    best_name, variant_table, ridge_alpha, lasso_alpha = choose_setup(base, dummy_cols)
    print(f"\n[eval] E2 variant selected on CV only -> {best_name}")
    print(variant_table.to_string(index=False))
    variant_table.to_csv(OUT_DIR / "E2_variant_selection_cv.csv", index=False)
    print(f"[eval] alphas selected on the same CV -> Ridge {ridge_alpha}, "
          f"Lasso {lasso_alpha}")
    models = build_models(ridge_alpha, lasso_alpha)

    print("\n[eval] E4 running rolling-origin backtest ...")
    bt = rolling_origin_backtest(base, best_name, models, dummy_cols)
    bt.to_csv(OUT_DIR / "E4_backtest_predictions.csv", index=False)

    summary = skill_vs_baseline(summarise_backtest(bt))
    summary.to_csv(OUT_DIR / "E4_backtest_metrics.csv", index=False)

    print("\n=== Forecast accuracy by horizon (assault rate per 100k) ===")
    print(summary[["Model", "Horizon_months", "MAE", "RMSE",
                   "MAPE_%", "Skill_%", "n"]].to_string(index=False,
                                                        float_format="%.3f"))

    shared = skill_vs_baseline(summarise_backtest(common_window(bt)))
    shared.to_csv(OUT_DIR / "E4_backtest_metrics_common_window.csv", index=False)
    print("\n=== Same table on the target months common to all horizons "
          f"({shared['First_target'].min()} to {shared['Last_target'].max()}) ===")
    print(shared.pivot(index="Model", columns="Horizon_months", values="RMSE")
          .round(1).to_string())

    # Best entry at each horizon, models and baselines alike
    winners = (summary.sort_values("RMSE").groupby("Horizon_months")
               .first()["Model"].to_dict())
    print("\n[eval] lowest RMSE by horizon:",
          ", ".join(f"h={h}: {m}" for h, m in winners.items()))

    # E10: best fitted model against the moving average, horizon by horizon
    fitted_names = list(models)
    rows = []
    for h in HORIZONS:
        at_h = summary[(summary["Horizon_months"] == h)
                       & summary["Model"].isin(fitted_names)]
        challenger = at_h.sort_values("RMSE").iloc[0]["Model"]
        rows.append({"Horizon_months": h, "Challenger": challenger,
                     "Reference": MOVING_AVERAGE,
                     **paired_bootstrap(bt[bt["h"] == h], challenger, MOVING_AVERAGE)})
    boot = pd.DataFrame(rows)
    boot.to_csv(OUT_DIR / "E4_backtest_paired_bootstrap.csv", index=False)
    print("\n=== E10 best fitted model vs 12-month moving average "
          "(RMSE difference, negative = model better) ===")
    print(boot.to_string(index=False, float_format="%.2f"))

    pi = prediction_intervals(bt, winners)
    pi.to_csv(OUT_DIR / "E5_prediction_intervals.csv", index=False)
    print("\n=== E5 80% prediction intervals for the best entry at each horizon ===")
    print(pi.to_string(index=False, float_format="%.1f"))

    plot_rmse_by_horizon(summary, PLOT_DIR / "R8_backtest_rmse_by_horizon.png")
    print(f"\n[eval] tables written to {OUT_DIR.name}/, "
          f"figure to {PLOT_DIR.name}/R8_backtest_rmse_by_horizon.png")


if __name__ == "__main__":
    main()
