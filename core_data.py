"""
core_data.py - Shared data caching, modeling, and utility module.
Used by both the Administrator Dashboard and Data Scientist Workbench.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_CSV = BASE_DIR / "dataset" / "processed" / "nt_crime_merged_2015_2025.csv"
RAW_POPULATION_CSV = BASE_DIR / "dataset" / "raw" / "population.csv"
MANIFEST_JSON = BASE_DIR / "dataset" / "raw" / "manifest.json"

EDA_PLOT_DIRS = [
    BASE_DIR / "Visualizations" / "eda_plots",
    BASE_DIR / "eda_plots",
]
REG_PLOT_DIRS = [
    BASE_DIR / "Visualizations" / "regression_plots",
    BASE_DIR / "regression_plots",
]

YEAR_MIN, YEAR_MAX = 2015, 2025
ALCOHOL_YEARS = (2023, 2024, 2025)
ASSAULT_CATEGORY = "02 Assault"

REGION_ORDER = [
    "Greater Darwin",
    "Central Australia",
    "Big Rivers",
    "East Arnhem",
    "Barkly",
    "Top End",
]

REGION_COLORS = {
    "Greater Darwin": "#1f77b4",
    "Central Australia": "#ff7f0e",
    "Big Rivers": "#2ca02c",
    "East Arnhem": "#d62728",
    "Barkly": "#9467bd",
    "Top End": "#8c564b",
}

REGION_GEO = {
    "Greater Darwin": {"lat": -12.4634, "lon": 130.8456, "centre": "Darwin / Palmerston"},
    "Central Australia": {"lat": -23.6980, "lon": 133.8807, "centre": "Alice Springs"},
    "Big Rivers": {"lat": -14.4646, "lon": 132.2635, "centre": "Katherine"},
    "East Arnhem": {"lat": -12.1825, "lon": 136.7808, "centre": "Nhulunbuy"},
    "Barkly": {"lat": -19.6465, "lon": 134.1916, "centre": "Tennant Creek"},
    "Top End": {"lat": -12.8500, "lon": 131.8500, "centre": "Top End Rural / Daly"},
}

KEY_TOWNS = [
    {"name": "Darwin", "region": "Greater Darwin", "lat": -12.4634, "lon": 130.8456, "role": "Territory Capital / Main Hospital"},
    {"name": "Palmerston", "region": "Greater Darwin", "lat": -12.4864, "lon": 130.9833, "role": "Satellite City Hub"},
    {"name": "Alice Springs", "region": "Central Australia", "lat": -23.6980, "lon": 133.8807, "role": "Regional Service Hub"},
    {"name": "Katherine", "region": "Big Rivers", "lat": -14.4646, "lon": 132.2635, "role": "Regional Police / Health Hub"},
    {"name": "Tennant Creek", "region": "Barkly", "lat": -19.6465, "lon": 134.1916, "role": "Remote Transit Hub"},
    {"name": "Nhulunbuy", "region": "East Arnhem", "lat": -12.1825, "lon": 136.7808, "role": "Mining & Community Hub"},
    {"name": "Jabiru", "region": "Top End", "lat": -12.6719, "lon": 132.8364, "role": "West Arnhem Centre"},
    {"name": "Wadeye (Thamarrurr)", "region": "Top End", "lat": -14.2375, "lon": 129.5208, "role": "Large Remote Community"},
    {"name": "Tiwi Islands (Wurrumiyanga)", "region": "Top End", "lat": -11.7584, "lon": 130.5898, "role": "Island Community Hub"},
    {"name": "Yuendumu", "region": "Central Australia", "lat": -22.2536, "lon": 131.7944, "role": "Anmatjere / Tanami Hub"},
    {"name": "Groote Eylandt (Alyangula)", "region": "East Arnhem", "lat": -13.8447, "lon": 136.4189, "role": "Mining Community Centre"},
]

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

FEATURE_VARIANTS_BASE = [
    "sin_month", "cos_month", "Season",
    "assault_rate_lag1", "assault_rate_lag3", "assault_rate_lag12"
]


@st.cache_data(show_spinner="Loading harmonised dataset...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed crime panel, raw population and monthly assault aggregates."""
    if not PROCESSED_CSV.exists():
        st.error(f"Processed dataset not found at {PROCESSED_CSV}. Please run Updated_End_to_End_pipeline.py first.")
        st.stop()

    panel = pd.read_csv(PROCESSED_CSV)
    
    if RAW_POPULATION_CSV.exists():
        pop_raw = pd.read_csv(RAW_POPULATION_CSV)
        pop_raw = pop_raw[pop_raw["Year"].between(YEAR_MIN, YEAR_MAX)].copy()
        total_pop = (pop_raw.groupby(["Year", "Region"], as_index=False)["Population"].sum()
                     .rename(columns={"Population": "Total_population"}))
        by_status = (pop_raw.groupby(["Year", "Region", "Aboriginal status"])["Population"]
                     .sum().unstack(fill_value=0).reset_index())
        by_status.columns.name = None
        by_sex = (pop_raw.groupby(["Year", "Region", "Sex"])["Population"]
                  .sum().unstack(fill_value=0).reset_index())
        by_sex.columns.name = None
        by_age = (pop_raw.groupby(["Year", "Region", "Age Group"])["Population"]
                  .sum().unstack(fill_value=0).reset_index())
        by_age.columns.name = None
        by_age = by_age.rename(columns={
            c: "Pop_age_" + str(c).replace("-", "").replace("+", "plus")
            for c in by_age.columns if c not in ("Year", "Region")
        })
        population = total_pop.merge(by_status, on=["Year", "Region"]).merge(by_sex, on=["Year", "Region"]).merge(by_age, on=["Year", "Region"])
    else:
        pop_cols = ["Year", "Region", "Total_population", "Aboriginal", "Non-Aboriginal", "Male", "Female"]
        pop_age_cols = [c for c in panel.columns if c.startswith("Pop_age_")]
        population = panel[pop_cols + pop_age_cols].drop_duplicates().reset_index(drop=True)

    assault = panel[panel["Offence category"] == ASSAULT_CATEGORY].copy()
    monthly_assault = (assault.groupby(["Year", "Quarter", "Month number", "Region"])
                       .agg(Assault_offences=("Number of offences", "sum"),
                            Alcohol_offences=("Alcohol_offences", "sum"),
                            DV_offences=("DV_offences", "sum"),
                            Total_PAC=("Total PAC", "first"),
                            Total_population=("Total_population", "first"),
                            Aboriginal=("Aboriginal", "first"))
                       .reset_index())
    monthly_assault["Assault_rate_100k"] = (monthly_assault["Assault_offences"]
                                            / monthly_assault["Total_population"]
                                            * 100_000).round(1)
    monthly_assault["alcohol_per_capita"] = (monthly_assault["Total_PAC"]
                                             / monthly_assault["Total_population"])

    return panel, population, monthly_assault


@st.cache_data(show_spinner="Preparing regression feature panel...")
def get_regression_panel(monthly_assault: pd.DataFrame, population: pd.DataFrame):
    """Mirror build_regression_panel & prepare_panel from pipeline."""
    obs = monthly_assault.copy()
    obs["date"] = pd.to_datetime(dict(year=obs["Year"], month=obs["Month number"], day=1))

    grid = pd.MultiIndex.from_product(
        [sorted(obs["Region"].unique()),
         pd.date_range(obs["date"].min(), obs["date"].max(), freq="MS")],
        names=["Region", "date"],
    ).to_frame(index=False)
    
    panel = grid.merge(
        obs.drop(columns=["Year", "Quarter", "Month number", "Total_population"]),
        on=["Region", "date"], how="left"
    )
    panel["Year"] = panel["date"].dt.year
    panel["Month number"] = panel["date"].dt.month
    panel["Quarter"] = panel["date"].dt.quarter
    panel = panel.merge(population[["Year", "Region", "Total_population"]],
                        on=["Year", "Region"], how="left")

    panel = panel.sort_values(["Region", "date"]).reset_index(drop=True)
    panel["log_assault_rate"] = np.log1p(panel["Assault_rate_100k"])
    panel["sin_month"] = np.sin(2 * np.pi * panel["Month number"] / 12)
    panel["cos_month"] = np.cos(2 * np.pi * panel["Month number"] / 12)
    panel["Season"] = panel["Month number"].isin([11, 12, 1, 2, 3, 4]).astype(int)  # 1=Wet

    for lag in (1, 3, 12):
        panel[f"assault_rate_lag{lag}"] = panel.groupby("Region")["Assault_rate_100k"].shift(lag)

    last = panel.loc[panel["Assault_offences"].notna(), "date"].max()
    panel["is_provisional"] = panel["date"] > (last - pd.DateOffset(months=6))

    dummies = pd.get_dummies(panel["Region"], prefix="Reg")
    dummies = dummies.drop(columns=["Reg_Greater Darwin"], errors="ignore")
    panel = pd.concat([panel, dummies], axis=1)

    required = ["log_assault_rate"] + [f"assault_rate_lag{l}" for l in (1, 3, 12)]
    panel = panel.dropna(subset=required).reset_index(drop=True)

    dummy_cols = [c for c in panel.columns if c.startswith("Reg_")]
    train = panel[panel["Year"] <= 2022].copy()
    test = panel[panel["Year"] >= 2023].copy()
    cv_frame = panel.sort_values(["date", "Region"]).reset_index(drop=True)

    return panel, train, test, dummy_cols, cv_frame


@st.cache_resource(show_spinner="Training predictive forecasting models...")
def run_model_training(train: pd.DataFrame, test: pd.DataFrame, cv_frame: pd.DataFrame, dummy_cols: list[str]):
    """Execute model selection, tuning, and evaluation identically to pipeline."""
    variants = {
        "V1: Temporal": FEATURE_VARIANTS_BASE,
        "V2: + Region": FEATURE_VARIANTS_BASE + dummy_cols,
        "V3: + Crime ctx": FEATURE_VARIANTS_BASE + ["Alcohol_offences", "DV_offences"],
        "V4: Full (no PAC)": (FEATURE_VARIANTS_BASE + ["Alcohol_offences", "DV_offences"] + dummy_cols),
    }

    cv_train = train.sort_values(["date", "Region"]).reset_index(drop=True)
    y_cv_train = cv_train["log_assault_rate"].to_numpy(float)
    tscv = TimeSeriesSplit(n_splits=5)

    variant_rmse = {}
    for name, feats in variants.items():
        X = cv_train[feats].astype(float).to_numpy()
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        scores = cross_val_score(pipe, X, y_cv_train, cv=tscv, scoring="neg_root_mean_squared_error")
        variant_rmse[name] = float(-scores.mean())

    best_variant = min(variant_rmse, key=variant_rmse.get)
    features = variants[best_variant].copy()

    # Backward elimination
    remaining = features.copy()
    while True:
        X = sm.add_constant(train[remaining].astype(float))
        y = train["log_assault_rate"].astype(float)
        ols_res = sm.OLS(y, X).fit()
        pvals = ols_res.pvalues.drop("const", errors="ignore")
        if pvals.empty:
            break
        worst_feature = pvals.idxmax()
        worst_p = pvals.max()
        if worst_p > 0.05:
            remaining.remove(worst_feature)
        else:
            break
    features = remaining

    # VIF
    X_with_const = sm.add_constant(train[features].astype(float))
    vif_records = []
    for i in range(1, X_with_const.shape[1]):
        val = variance_inflation_factor(X_with_const.to_numpy(), i)
        feat = features[i - 1]
        vif_records.append({"Feature": feat, "VIF": val, "Status": "Remove" if val > 30 else ("Monitor" if val > 10 else "Acceptable")})
    vif_df = pd.DataFrame(vif_records).sort_values("VIF", ascending=False)
    to_remove = vif_df.loc[vif_df["VIF"] > 30, "Feature"].tolist()
    features = [f for f in features if f not in to_remove]

    # Alpha tuning
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_cv = {}
    lasso_cv = {}
    y_cv_all = cv_frame["log_assault_rate"].to_numpy(float)
    X_cv_all = cv_frame[features].astype(float).to_numpy()

    for a in alphas:
        pipe_r = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=a))])
        pipe_l = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=a, max_iter=10000))])
        ridge_cv[a] = float(-cross_val_score(pipe_r, X_cv_all, y_cv_all, cv=tscv, scoring="neg_root_mean_squared_error").mean())
        lasso_cv[a] = float(-cross_val_score(pipe_l, X_cv_all, y_cv_all, cv=tscv, scoring="neg_root_mean_squared_error").mean())

    best_ridge = min(ridge_cv, key=ridge_cv.get)
    best_lasso = min(lasso_cv, key=lasso_cv.get)

    # XGBoost tuning
    param_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "reg_alpha": 1.0, "reg_lambda": 3.0},
        {"max_depth": 4, "learning_rate": 0.03, "reg_alpha": 2.0, "reg_lambda": 4.0},
        {"max_depth": 5, "learning_rate": 0.02, "reg_alpha": 3.0, "reg_lambda": 5.0},
        {"max_depth": 6, "learning_rate": 0.02, "reg_alpha": 4.0, "reg_lambda": 6.0},
    ]
    xgb_scores = {}
    for idx, p in enumerate(param_grid):
        m = XGBRegressor(n_estimators=800, subsample=0.9, colsample_bytree=0.9, min_child_weight=1,
                         objective="reg:squarederror", random_state=42, **p)
        score = float(-cross_val_score(m, X_cv_all, y_cv_all, cv=tscv, scoring="neg_root_mean_squared_error").mean())
        xgb_scores[idx] = score
    best_xgb_idx = min(xgb_scores, key=xgb_scores.get)
    best_xgb_params = param_grid[best_xgb_idx]

    # Model A Fitting
    y_train = train["log_assault_rate"].to_numpy(float)
    y_test = test["log_assault_rate"].to_numpy(float)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(train[features].astype(float))
    x_test_scaled = scaler.transform(test[features].astype(float))

    X_train_raw = train[features].astype(float)
    X_test_raw = test[features].astype(float)

    models = {
        "Linear Regression": LinearRegression(),
        f"Ridge (α={best_ridge})": Ridge(alpha=best_ridge),
        f"Lasso (α={best_lasso})": Lasso(alpha=best_lasso, max_iter=10000),
        "XGBoost": XGBRegressor(n_estimators=800, subsample=0.9, colsample_bytree=0.9, min_child_weight=1,
                                objective="reg:squarederror", random_state=42, **best_xgb_params),
    }

    results = {}
    for name, model in models.items():
        if name == "XGBoost":
            model.fit(X_train_raw, y_train)
            pred = model.predict(X_test_raw)
            cv_val = xgb_scores[best_xgb_idx]
        else:
            model.fit(x_train_scaled, y_train)
            pred = model.predict(x_test_scaled)
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model.__class__(**model.get_params()))])
            cv_val = float(-cross_val_score(pipe, X_cv_all, y_cv_all, cv=tscv, scoring="neg_root_mean_squared_error").mean())

        rmse_log = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae_log = float(mean_absolute_error(y_test, pred))
        rmse_rate = float(np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred))))

        results[name] = {
            "model": model,
            "pred": pred,
            "rmse": rmse_log,
            "mae": mae_log,
            "rmse_rate": rmse_rate,
            "cv": cv_val,
        }

    best_name = min(results, key=lambda k: results[k]["rmse"])

    # Model B
    panel_b = cv_frame[(cv_frame["Year"] >= min(ALCOHOL_YEARS)) & cv_frame["Total_PAC"].notna()].copy()
    panel_b["alcohol_per_capita"] = panel_b["Total_PAC"] / panel_b["Total_population"]
    features_b = features + ["alcohol_per_capita"]
    train_b = panel_b[panel_b["Year"] <= 2024]
    test_b = panel_b[panel_b["Year"] == YEAR_MAX]

    results_b = {}
    if len(train_b) > 5 and not test_b.empty:
        y_train_b = train_b["log_assault_rate"].to_numpy(float)
        y_test_b = test_b["log_assault_rate"].to_numpy(float)

        scaler_b = StandardScaler()
        x_tr_b = scaler_b.fit_transform(train_b[features_b].astype(float))
        x_te_b = scaler_b.transform(test_b[features_b].astype(float))

        lin_b = {
            "Linear Regression": LinearRegression(),
            f"Ridge (α={best_ridge})": Ridge(alpha=best_ridge),
            f"Lasso (α={best_lasso})": Lasso(alpha=best_lasso, max_iter=10000),
        }
        for n, m in lin_b.items():
            m.fit(x_tr_b, y_train_b)
            p = m.predict(x_te_b)
            results_b[n] = {
                "rmse": float(np.sqrt(mean_squared_error(y_test_b, p))),
                "mae": float(mean_absolute_error(y_test_b, p)),
                "rmse_rate": float(np.sqrt(mean_squared_error(np.expm1(y_test_b), np.expm1(p)))),
                "coef": dict(zip(features_b, m.coef_)),
            }

        xgb_b = XGBRegressor(n_estimators=800, subsample=0.9, colsample_bytree=0.9, min_child_weight=1,
                             objective="reg:squarederror", random_state=42, **best_xgb_params)
        xgb_b.fit(train_b[features_b].astype(float), y_train_b)
        p_xgb = xgb_b.predict(test_b[features_b].astype(float))
        results_b["XGBoost"] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test_b, p_xgb))),
            "mae": float(mean_absolute_error(y_test_b, p_xgb)),
            "rmse_rate": float(np.sqrt(mean_squared_error(np.expm1(y_test_b), np.expm1(p_xgb)))),
            "coef": None,
        }

    return {
        "variant_rmse": variant_rmse,
        "best_variant": best_variant,
        "features": features,
        "vif_df": vif_df,
        "ridge_cv": ridge_cv,
        "lasso_cv": lasso_cv,
        "best_ridge": best_ridge,
        "best_lasso": best_lasso,
        "xgb_scores": xgb_scores,
        "param_grid": param_grid,
        "best_xgb_params": best_xgb_params,
        "results": results,
        "best_name": best_name,
        "scaler": scaler,
        "results_b": results_b,
        "features_b": features_b if "features_b" in locals() else [],
        "test_b": test_b if "test_b" in locals() else pd.DataFrame(),
    }
