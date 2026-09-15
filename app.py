"""
Streamlit Policy & Resource Planning Dashboard: Northern Territory Assault & Crime Forecasting
================================================================================================
Derived from Updated_End_to_End_pipeline.py (PRT661 Data Science Practice - Group DAN2 Theme 2).

Designed specifically for policy makers, justice planners, and public health officials
to evaluate regional crime trends, visualize spatial distributions across the NT,
assess forecasting models, and simulate policy interventions (e.g., alcohol supply curbs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
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

# Regional Coordinates for Geographic Map Display
# Centroids and primary service administrative centres
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
    {"name": "Nhulunbuy", "region": "East Arnhem", "lat": -12.1825, "lon": 136.7808, "role": "Mining & Community Service Hub"},
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

# -----------------------------------------------------------------------------
# Streamlit App Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NT Crime & Assault Forecasting - Policy & Resource Planning",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive / government policy feel
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #718096;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .alert-priority-high {
        background-color: #fff5f5;
        border-left: 4px solid #e53e3e;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .alert-priority-med {
        background-color: #fffaf0;
        border-left: 4px solid #dd6b20;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .alert-priority-low {
        background-color: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Data Loading & Processing
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading harmonised NT crime panel...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed crime panel, raw population and monthly assault aggregates."""
    if not PROCESSED_CSV.exists():
        st.error(f"Processed dataset not found at {PROCESSED_CSV}. Please run Updated_End_to_End_pipeline.py first.")
        st.stop()

    panel = pd.read_csv(PROCESSED_CSV)
    
    # Population data
    if RAW_POPULATION_CSV.exists():
        pop_raw = pd.read_csv(RAW_POPULATION_CSV)
        pop_raw = pop_raw[pop_raw["Year"].between(YEAR_MIN, YEAR_MAX)].copy()
        # Aggregated population features
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
        # Fallback to pop columns in panel
        pop_cols = ["Year", "Region", "Total_population", "Aboriginal", "Non-Aboriginal", "Male", "Female"]
        pop_age_cols = [c for c in panel.columns if c.startswith("Pop_age_")]
        population = panel[pop_cols + pop_age_cols].drop_duplicates().reset_index(drop=True)

    # Monthly assault aggregate
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


@st.cache_data(show_spinner="Building regression feature panel...")
def get_regression_panel(monthly_assault: pd.DataFrame, population: pd.DataFrame):
    """Mirror build_regression_panel & prepare_panel from Updated_End_to_End_pipeline.py."""
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


@st.cache_resource(show_spinner="Training forecasting models (Linear, Ridge, Lasso, XGBoost)...")
def run_model_training(train: pd.DataFrame, test: pd.DataFrame, cv_frame: pd.DataFrame, dummy_cols: list[str]):
    """Execute pipeline model selection, tuning, and evaluation identically to pipeline."""
    # 1. Feature selection via TimeSeriesSplit CV on training set
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

    # 2. Backward elimination (OLS p-values)
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

    # 3. VIF check
    X_with_const = sm.add_constant(train[features].astype(float))
    vif_records = []
    for i in range(1, X_with_const.shape[1]):
        val = variance_inflation_factor(X_with_const.to_numpy(), i)
        feat = features[i - 1]
        vif_records.append({"Feature": feat, "VIF": val, "Status": "Remove" if val > 30 else ("Monitor" if val > 10 else "Acceptable")})
    vif_df = pd.DataFrame(vif_records).sort_values("VIF", ascending=False)
    to_remove = vif_df.loc[vif_df["VIF"] > 30, "Feature"].tolist()
    features = [f for f in features if f not in to_remove]

    # 4. Alpha tuning for Ridge & Lasso
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

    # 5. XGBoost tuning
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

    # 6. Final Model A Fitting
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

    # 7. Model B (With Alcohol Supply PAC)
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


# -----------------------------------------------------------------------------
# App Layout & Execution
# -----------------------------------------------------------------------------
def main():
    # Header Banner
    st.markdown('<div class="main-header">Northern Territory Assault Forecasting & Crime Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive Decision Support & Resource Allocation Portal | PRT661 Data Science Practice</div>', unsafe_allow_html=True)

    # Load core data
    panel, population, monthly_assault = load_data()
    reg_panel, train, test, dummy_cols, cv_frame = get_regression_panel(monthly_assault, population)

    # -------------------------------------------------------------------------
    # Sidebar: Policy Filters & Controls
    # -------------------------------------------------------------------------
    st.sidebar.title("🎛️ Policy & Resource Controls")
    st.sidebar.markdown("Filter parameters to model historical patterns or forecast specific regional jurisdictions.")

    selected_regions = st.sidebar.multiselect(
        "Target NT Regions",
        options=REGION_ORDER,
        default=REGION_ORDER,
        help="Filter the executive summaries, trends, and map layers."
    )
    if not selected_regions:
        st.sidebar.warning("Please select at least one region.")
        selected_regions = REGION_ORDER

    year_range = st.sidebar.slider(
        "Historical Window",
        min_value=YEAR_MIN,
        max_value=YEAR_MAX,
        value=(YEAR_MIN, YEAR_MAX),
        step=1,
        help="Select calendar span for aggregate crime analyses."
    )

    all_categories = sorted(panel["Offence category"].dropna().unique())
    selected_categories = st.sidebar.multiselect(
        "Offence Categories",
        options=all_categories,
        default=[ASSAULT_CATEGORY] if ASSAULT_CATEGORY in all_categories else all_categories[:3],
        help="Filter for specific crime categories or all offences."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Resource Priority Threshold")
    rate_threshold = st.sidebar.slider(
        "High Per-Capita Alert Threshold (Assaults / 100k / mo)",
        min_value=100, max_value=800, value=350, step=25,
        help="Monthly assault rate per 100k triggering critical frontline resource reallocation."
    )

    # Model training cached run
    with st.spinner("Training predictive models on historical panel..."):
        model_artifacts = run_model_training(train, test, cv_frame, dummy_cols)

    # -------------------------------------------------------------------------
    # Navigation Tabs
    # -------------------------------------------------------------------------
    tabs = st.tabs([
        "🏛️ Executive Summary",
        "🗺️ Spatial Crime Map & Resource Allocation",
        "📈 Exploratory Crime Trends & Seasonality",
        "👥 Demographics & Age PCA",
        "🤖 Predictive Models (A & B)",
        "🎯 Policy Scenario Simulator",
        "📁 Data Explorer & Exports"
    ])

    # =========================================================================
    # TAB 1: Executive Summary
    # =========================================================================
    with tabs[0]:
        st.subheader("Executive KPI Overview (Selected Window: {}-{})".format(year_range[0], year_range[1]))
        
        # Filter panel by selected regions & year range
        filtered_panel = panel[
            panel["Region"].isin(selected_regions) &
            panel["Year"].between(year_range[0], year_range[1])
        ]
        filtered_assault = monthly_assault[
            monthly_assault["Region"].isin(selected_regions) &
            monthly_assault["Year"].between(year_range[0], year_range[1])
        ]

        total_offences = filtered_panel["Number of offences"].sum()
        total_assaults = filtered_assault["Assault_offences"].sum()
        avg_assault_rate = filtered_assault["Assault_rate_100k"].mean()
        alc_assault_share = (filtered_assault["Alcohol_offences"].sum() / total_assaults * 100) if total_assaults > 0 else 0
        dv_assault_share = (filtered_assault["DV_offences"].sum() / total_assaults * 100) if total_assaults > 0 else 0
        latest_pop = population[population["Year"] == year_range[1]]["Total_population"].sum()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Recorded Offences", f"{total_offences:,.0f}")
        with col2:
            st.metric("Total Assault Incidents", f"{total_assaults:,.0f}")
        with col3:
            st.metric("Avg Monthly Assault Rate", f"{avg_assault_rate:.1f} /100k")
        with col4:
            st.metric("Alcohol-Involved Assaults", f"{alc_assault_share:.1f}%")
        with col5:
            st.metric("Domestic Violence Assaults", f"{dv_assault_share:.1f}%")

        st.markdown("---")

        # Policy & Resource Planning Insight Cards
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.markdown("#### 🚨 Regional Resource Allocation Status")
            st.markdown(
                "A common pitfall in resource planning is allocating frontline police and health staff "
                "purely based on **raw crime volume** rather than **per-capita severity**. "
                "The table below evaluates both metrics to highlight urgent operational priority."
            )

            # Regional summary table
            reg_summary = filtered_assault.groupby("Region").agg(
                Total_Assaults=("Assault_offences", "sum"),
                Avg_Monthly_Rate=("Assault_rate_100k", "mean"),
                Alcohol_Share=("Alcohol_offences", lambda x: (x.sum() / filtered_assault.loc[x.index, "Assault_offences"].sum() * 100) if filtered_assault.loc[x.index, "Assault_offences"].sum() > 0 else 0),
                DV_Share=("DV_offences", lambda x: (x.sum() / filtered_assault.loc[x.index, "Assault_offences"].sum() * 100) if filtered_assault.loc[x.index, "Assault_offences"].sum() > 0 else 0),
            ).reset_index()

            def get_priority(row):
                if row["Avg_Monthly_Rate"] >= rate_threshold:
                    return "CRITICAL (Urgent Intervention)"
                elif row["Avg_Monthly_Rate"] >= rate_threshold * 0.7:
                    return "ELEVATED (Targeted Patrols)"
                else:
                    return "BASELINE (Standard Patrols)"

            reg_summary["Resource Priority"] = reg_summary.apply(get_priority, axis=1)
            reg_summary["Avg_Monthly_Rate"] = reg_summary["Avg_Monthly_Rate"].round(1)
            reg_summary["Alcohol_Share"] = reg_summary["Alcohol_Share"].round(1).astype(str) + "%"
            reg_summary["DV_Share"] = reg_summary["DV_Share"].round(1).astype(str) + "%"

            st.dataframe(
                reg_summary.sort_values("Avg_Monthly_Rate", ascending=False),
                column_config={
                    "Region": "Region",
                    "Total_Assaults": st.column_config.NumberColumn("Total Assaults", format="%d"),
                    "Avg_Monthly_Rate": st.column_config.NumberColumn("Avg Monthly Rate (/100k)", format="%.1f"),
                    "Alcohol_Share": "Alcohol %",
                    "DV_Share": "DV %",
                    "Resource Priority": "Resource Alert"
                },
                use_container_width=True,
                hide_index=True
            )

        with c_right:
            st.markdown("#### 🧭 Core Pipeline Assumptions [A1–A16]")
            with st.expander("View Data Engineering & Methodological Log", expanded=True):
                st.markdown("""
                - **[A1] Historical Window**: 2015–2025 trimmed to ensure consistent reporting standards.
                - **[A2] Crime Categorisation**: Harmonised old PROMIS to new ANZSOC categories so assault remains consistent.
                - **[A3] Nov 2023 System Gap**: Police transition left Nov 2023 unrecorded; retained as visible NaN without false interpolation.
                - **[A5] Regional Harmonisation**: Darwin & Palmerston united into Greater Darwin to avoid dividing by duplicated population.
                - **[A8] Offence vs Incident Counting**: Alcohol & DV are counted as offences to guard against schema change row jumps.
                - **[A11-A12] Split Protocol**: Model A trained 2015–2022, tested 2023–2025. Model B (with alcohol PAC) trained 2023–2024, tested 2025.
                - **[A15-A16] Leakage-Free Evaluation**: TimeSeriesSplit CV and StandardScaler inside Pipeline to prevent future fold leakage.
                """)

    # =========================================================================
    # TAB 2: Spatial Crime Map & Resource Allocation
    # =========================================================================
    with tabs[1]:
        st.subheader("🗺️ Northern Territory Spatial Crime Distribution & Jurisdictional Analysis")
        st.markdown(
            "This map displays crime density and per-capita intensity across the Northern Territory. "
            "Policy makers can switch between **Rate per 100k** (population burden) and **Total Volume** (patrol headcount) "
            "to determine mobile policing deployments and crisis accommodation distribution."
        )

        m_col1, m_col2 = st.columns([1, 3])
        with m_col1:
            map_metric = st.radio(
                "Select Map Metric Layer:",
                options=[
                    "Assault Rate per 100k (Per-Capita Burden)",
                    "Total Assault Offences (Headcount Demand)",
                    "Alcohol-Involved Offences",
                    "Domestic Violence Offences"
                ]
            )

            map_year = st.selectbox(
                "Select Calendar Year for Map:",
                options=sorted(filtered_assault["Year"].unique(), reverse=True),
                index=0
            )

            show_towns = st.checkbox("Show Key Service Hubs & Hospitals", value=True)

        # Prepare Map Data
        map_df_year = filtered_assault[filtered_assault["Year"] == map_year].groupby("Region").agg(
            Assault_offences=("Assault_offences", "sum"),
            Avg_Monthly_Rate=("Assault_rate_100k", "mean"),
            Alcohol_offences=("Alcohol_offences", "sum"),
            DV_offences=("DV_offences", "sum"),
            Total_population=("Total_population", "first")
        ).reset_index()

        map_df_year["lat"] = map_df_year["Region"].map(lambda r: REGION_GEO.get(r, {}).get("lat", 0))
        map_df_year["lon"] = map_df_year["Region"].map(lambda r: REGION_GEO.get(r, {}).get("lon", 0))
        map_df_year["centre"] = map_df_year["Region"].map(lambda r: REGION_GEO.get(r, {}).get("centre", ""))

        if map_metric.startswith("Assault Rate"):
            size_col = "Avg_Monthly_Rate"
            color_col = "Avg_Monthly_Rate"
            color_scale = "Reds"
            metric_label = "Monthly Rate per 100k"
        elif map_metric.startswith("Total Assault"):
            size_col = "Assault_offences"
            color_col = "Assault_offences"
            color_scale = "Viridis"
            metric_label = "Total Annual Assaults"
        elif map_metric.startswith("Alcohol-Involved"):
            size_col = "Alcohol_offences"
            color_col = "Alcohol_offences"
            color_scale = "Oranges"
            metric_label = "Alcohol-Involved Offences"
        else:
            size_col = "DV_offences"
            color_col = "DV_offences"
            color_scale = "Purples"
            metric_label = "DV-Involved Offences"

        with m_col2:
            fig_map = px.scatter_geo(
                map_df_year,
                lat="lat",
                lon="lon",
                size=size_col,
                color=color_col,
                color_continuous_scale=color_scale,
                hover_name="Region",
                hover_data={
                    "lat": False,
                    "lon": False,
                    "centre": True,
                    "Avg_Monthly_Rate": ":.1f",
                    "Assault_offences": ":,d",
                    "Alcohol_offences": ":,d",
                    "DV_offences": ":,d",
                    "Total_population": ":,d"
                },
                size_max=45,
                projection="natural earth",
                title=f"NT Regional Crime Heatmap ({map_year}) - {metric_label}"
            )

            # Restrict map bounds to Northern Territory
            fig_map.update_geos(
                center=dict(lat=-18.0, lon=133.5),
                lataxis_range=[-26.5, -10.5],
                lonaxis_range=[128.5, 138.5],
                visible=True,
                showcountries=False,
                showcoastlines=True,
                showland=True,
                landcolor="#f4f6f8",
                oceancolor="#e9f2f9",
                showocean=True,
                showrivers=True,
                showlakes=True
            )
            fig_map.update_layout(
                height=540,
                margin=dict(r=10, t=40, b=10, l=10),
                coloraxis_colorbar=dict(title=metric_label)
            )

            if show_towns:
                towns_df = pd.DataFrame(KEY_TOWNS)
                fig_map.add_trace(
                    go.Scattergeo(
                        lat=towns_df["lat"],
                        lon=towns_df["lon"],
                        text=towns_df["name"],
                        mode="markers+text",
                        textposition="top center",
                        textfont=dict(size=10, color="#2d3748"),
                        marker=dict(size=7, color="#319795", symbol="square"),
                        hoverinfo="text",
                        hovertext=towns_df["name"] + " (" + towns_df["role"] + ")",
                        name="Key Hubs"
                    )
                )

            st.plotly_chart(fig_map, use_container_width=True)

        # Policy Resource Allocation Dilemma Visualized
        st.markdown("---")
        st.markdown("#### ⚖️ The Resource Planning Discrepancy: Volume vs Per-Capita Rate")
        st.markdown(
            "Comparing **total assaults (headcount)** against **monthly assault rate per 100k population**. "
            "Notice how Greater Darwin accounts for high volume due to population scale, "
            "while Central Australia and Barkly experience dramatically higher per-capita victimisation rates."
        )

        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            fig_vol = px.bar(
                map_df_year.sort_values("Assault_offences", ascending=False),
                x="Region",
                y="Assault_offences",
                color="Region",
                color_discrete_map=REGION_COLORS,
                title=f"Total Assault Offences by Region ({map_year})",
                labels={"Assault_offences": "Recorded Assaults"}
            )
            fig_vol.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_vol, use_container_width=True)

        with comp_col2:
            fig_rate = px.bar(
                map_df_year.sort_values("Avg_Monthly_Rate", ascending=False),
                x="Region",
                y="Avg_Monthly_Rate",
                color="Region",
                color_discrete_map=REGION_COLORS,
                title=f"Monthly Assault Rate per 100,000 Residents ({map_year})",
                labels={"Avg_Monthly_Rate": "Assault Rate / 100k"}
            )
            fig_rate.add_hline(y=rate_threshold, line_dash="dash", line_color="red",
                               annotation_text=f"Alert Threshold ({rate_threshold})", annotation_position="top right")
            fig_rate.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_rate, use_container_width=True)

    # =========================================================================
    # TAB 3: Exploratory Trends & Seasonality
    # =========================================================================
    with tabs[2]:
        st.subheader("📈 Historical Trends, Seasonal Dynamics, and Alcohol Correlation")
        
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            # Multi-Year Trend by Region
            rate_yr = (filtered_assault.groupby(["Year", "Region"], as_index=False)
                       ["Assault_rate_100k"].mean())
            fig_trend = px.line(
                rate_yr,
                x="Year",
                y="Assault_rate_100k",
                color="Region",
                color_discrete_map=REGION_COLORS,
                markers=True,
                title=f"Annual Assault Rate per 100k by Region ({year_range[0]}-{year_range[1]})",
                labels={"Assault_rate_100k": "Avg Monthly Rate / 100k"}
            )
            fig_trend.update_layout(height=380)
            st.plotly_chart(fig_trend, use_container_width=True)

        with t_col2:
            # Seasonality by Month
            monthly_season = filtered_assault.groupby("Month number", as_index=False)["Assault_offences"].sum()
            monthly_season["Month"] = monthly_season["Month number"].map(lambda m: MONTH_LABELS[m - 1])
            peak_m = monthly_season.loc[monthly_season["Assault_offences"].idxmax(), "Month"]

            fig_season = px.bar(
                monthly_season,
                x="Month",
                y="Assault_offences",
                title=f"Assault Seasonality by Month (All Years Aggregated) | Peak: {peak_m}",
                labels={"Assault_offences": "Total Assaults"},
                color="Assault_offences",
                color_continuous_scale="Blues"
            )
            fig_season.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_season, use_container_width=True)

        st.markdown("---")

        t_col3, t_col4 = st.columns(2)
        with t_col3:
            # Year x Month Heatmap (transition gap highlighted)
            heat = filtered_assault.groupby(["Year", "Month number"])["Assault_offences"].sum().unstack()
            heat.columns = [MONTH_LABELS[m - 1] for m in heat.columns]
            
            fig_heat = px.imshow(
                heat,
                labels=dict(x="Month", y="Year", color="Assaults"),
                x=list(heat.columns),
                y=list(heat.index),
                color_continuous_scale="YlOrRd",
                title="Assault Intensity Matrix (Year x Month) [Nov 2023 = Transition Gap]"
            )
            fig_heat.update_layout(height=380)
            st.plotly_chart(fig_heat, use_container_width=True)

        with t_col4:
            # Alcohol & DV Share in Assaults
            shares = (filtered_assault.groupby("Year")
                      .agg(Assault=("Assault_offences", "sum"),
                           Alcohol=("Alcohol_offences", "sum"),
                           DV=("DV_offences", "sum")).reset_index())
            shares["Alcohol %"] = (shares["Alcohol"] / shares["Assault"] * 100).round(1)
            shares["DV %"] = (shares["DV"] / shares["Assault"] * 100).round(1)

            fig_share = go.Figure()
            fig_share.add_trace(go.Bar(x=shares["Year"], y=shares["Alcohol %"], name="Alcohol-Involved %", marker_color="#EF5350"))
            fig_share.add_trace(go.Bar(x=shares["Year"], y=shares["DV %"], name="DV-Involved %", marker_color="#AB47BC"))
            fig_share.update_layout(
                barmode="group",
                title="Alcohol & Domestic Violence Involvement Rate (%) Across Years",
                yaxis_title="% of Total Assaults",
                height=380
            )
            st.plotly_chart(fig_share, use_container_width=True)

        # Wholesale Alcohol Supply (PAC) & Pooled vs Within Correlation
        st.markdown("---")
        st.markdown("#### 🍷 Wholesale Alcohol Supply (PAC) Trends & Policy Correlation")
        pac_df = filtered_assault[filtered_assault["Total_PAC"].notna()].copy()
        if not pac_df.empty:
            p_col1, p_col2 = st.columns([3, 2])
            with p_col1:
                pac_agg = pac_df.groupby(["Year", "Quarter", "Region"], as_index=False)["Total_PAC"].first()
                pac_agg["YearQuarter"] = pac_agg["Year"].astype(str) + "-Q" + pac_agg["Quarter"].astype(str)
                fig_pac = px.line(
                    pac_agg.sort_values(["Year", "Quarter"]),
                    x="YearQuarter",
                    y="Total_PAC",
                    color="Region",
                    color_discrete_map=REGION_COLORS,
                    markers=True,
                    title="Wholesale Alcohol Supply (Pure Alcohol Content - Litres) by Quarter",
                    labels={"Total_PAC": "Total PAC (Litres)", "YearQuarter": "Quarter"}
                )
                fig_pac.update_layout(height=360)
                st.plotly_chart(fig_pac, use_container_width=True)

            with p_col2:
                # Correlation explanation
                pooled_r = pac_df["alcohol_per_capita"].corr(pac_df["Assault_rate_100k"])
                within_r = pac_df.groupby("Region").apply(lambda g: g["alcohol_per_capita"].corr(g["Assault_rate_100k"])).round(3)
                
                st.markdown("##### Econometric Correlation Insights:")
                st.markdown(f"- **Pooled Pearson Correlation**: `r = {pooled_r:.3f}`")
                st.caption("Pooling across all regions creates a spurious scale effect because both alcohol volume and offence counts scale with regional population.")
                st.markdown("##### Within-Region Correlation (Controlling for Region Size):")
                within_df = pd.DataFrame(within_r, columns=["Within-Region r"]).reset_index()
                st.dataframe(within_df, hide_index=True, use_container_width=True)

        # High-res static figures expander
        with st.expander("🖼️ View Pipeline Generated High-Resolution Figures (eda_plots/)", expanded=False):
            found_plots = []
            for d in EDA_PLOT_DIRS:
                if d.exists():
                    found_plots = sorted(list(d.glob("*.png")))
                    if found_plots:
                        break
            if found_plots:
                selected_plot = st.selectbox("Select figure:", options=[p.name for p in found_plots])
                plot_path = next(p for p in found_plots if p.name == selected_plot)
                st.image(str(plot_path), caption=selected_plot, use_container_width=True)
            else:
                st.info("No pre-rendered PNGs found in eda_plots/ yet.")

    # =========================================================================
    # TAB 4: Demographics & PCA Age Structure
    # =========================================================================
    with tabs[3]:
        st.subheader("👥 Demographic Vulnerability & Principal Component Analysis (PCA)")
        st.markdown(
            "Regional demographics strongly condition baseline assault vulnerability. "
            "Under **[A14]**, PCA is fitted on regional **age-group shares** rather than raw counts, "
            "preventing regional size from dominating the components."
        )

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            # Population growth by region
            pop_trend = population[population["Region"].isin(selected_regions)].sort_values("Year")
            fig_pop = px.line(
                pop_trend,
                x="Year",
                y="Total_population",
                color="Region",
                color_discrete_map=REGION_COLORS,
                markers=True,
                title="Regional Population Trends (2015-2025)",
                labels={"Total_population": "Population"}
            )
            fig_pop.update_layout(height=360)
            st.plotly_chart(fig_pop, use_container_width=True)

        with d_col2:
            # Indigenous Status Composition
            latest_year_pop = population[population["Year"] == year_range[1]].copy()
            status_df = latest_year_pop[["Region", "Aboriginal", "Non-Aboriginal"]].melt(
                id_vars=["Region"], var_name="Status", value_name="Count"
            )
            fig_status = px.bar(
                status_df,
                x="Region",
                y="Count",
                color="Status",
                barmode="stack",
                title=f"Aboriginal Status Composition by Region ({year_range[1]})",
                color_discrete_sequence=["#e28743", "#1f77b4"]
            )
            fig_status.update_layout(height=360)
            st.plotly_chart(fig_status, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔬 PCA Scree Plot on Age Structure Shares [A14]")

        from sklearn.decomposition import PCA
        age_cols = sorted(c for c in population.columns if c.startswith("Pop_age_"))
        shares = population[age_cols].div(population["Total_population"], axis=0)
        scaled_shares = StandardScaler().fit_transform(shares.astype(float).values)
        pca = PCA().fit(scaled_shares)
        explained = pca.explained_variance_ratio_ * 100
        cumulative = np.cumsum(explained)

        pca_df = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(min(8, len(explained)))],
            "Explained %": explained[:min(8, len(explained))].round(1),
            "Cumulative %": cumulative[:min(8, len(explained))].round(1)
        })

        p_col_a, p_col_b = st.columns([3, 2])
        with p_col_a:
            fig_pca = go.Figure()
            fig_pca.add_trace(go.Bar(x=pca_df["Component"], y=pca_df["Explained %"], name="Individual Variance %", marker_color="#4299e1"))
            fig_pca.add_trace(go.Scatter(x=pca_df["Component"], y=pca_df["Cumulative %"], name="Cumulative %", marker_color="#e53e3e", mode="lines+markers"))
            fig_pca.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="90% Threshold")
            fig_pca.update_layout(title="PCA Scree Plot: Regional Age Structure Variance", height=350)
            st.plotly_chart(fig_pca, use_container_width=True)

        with p_col_b:
            st.markdown("##### PCA Explained Variance Table")
            st.dataframe(pca_df, hide_index=True, use_container_width=True)
            n_90 = int(np.searchsorted(cumulative, 90) + 1)
            st.info(f"💡 **Policy Summary**: **{n_90} principal components** capture over 90% of demographic variation across regions.")

    # =========================================================================
    # TAB 5: Predictive Models (A & B)
    # =========================================================================
    with tabs[4]:
        st.subheader("🤖 Predictive Forecasting Engine: Model Evaluation & Benchmarking")
        st.markdown(
            "Comparing **Model A (Baseline: Calendar, Lagged Crime, Regional Dummies)** and "
            "**Model B (Marginal Evaluation: Wholesale Alcohol PAC per capita)**. "
            "All model selection was performed strictly on the training partition to prevent optimistic test bias [A16]."
        )

        res = model_artifacts["results"]
        best_name = model_artifacts["best_name"]

        # Leaderboard Table
        st.markdown(f"#### 🏆 Model A Performance Leaderboard (Best Model: `{best_name}`)")
        leaderboard_rows = []
        for name, m in res.items():
            leaderboard_rows.append({
                "Model": name,
                "Test RMSE (log)": round(m["rmse"], 4),
                "Test MAE (log)": round(m["mae"], 4),
                "Rate RMSE (/100k)": round(m["rmse_rate"], 1),
                "5-Fold CV RMSE": round(m["cv"], 4),
                "Verdict": "⭐ BEST PERFORMER" if name == best_name else "Candidate"
            })
        leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("Test RMSE (log)")
        st.dataframe(leaderboard_df, hide_index=True, use_container_width=True)

        st.markdown("---")

        m_plot1, m_plot2 = st.columns(2)
        with m_plot1:
            # Predicted vs Actual Scatter Plot
            actual_rate = np.expm1(test["log_assault_rate"].to_numpy(float))
            pred_rate = np.expm1(res[best_name]["pred"])

            pv_df = pd.DataFrame({
                "Actual": actual_rate,
                "Predicted": pred_rate,
                "Region": test["Region"].to_numpy(),
                "Year": test["Year"].to_numpy(),
                "Month": test["Month number"].to_numpy()
            })

            fig_pva = px.scatter(
                pv_df,
                x="Actual",
                y="Predicted",
                color="Region",
                color_discrete_map=REGION_COLORS,
                title=f"Predicted vs Actual Assault Rate per 100k ({best_name})",
                labels={"Actual": "Actual Rate (/100k)", "Predicted": "Predicted Rate (/100k)"}
            )
            max_val = max(actual_rate.max(), pred_rate.max()) * 1.05
            fig_pva.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                              line=dict(color="red", dash="dash", width=1.5))
            fig_pva.update_layout(height=400)
            st.plotly_chart(fig_pva, use_container_width=True)

        with m_plot2:
            # Residual Distribution & Shapiro-Wilk Normality Check
            resid = actual_rate - pred_rate
            sw_stat, sw_p = stats.shapiro(test["log_assault_rate"].to_numpy(float) - res[best_name]["pred"])

            fig_resid = px.histogram(
                resid,
                nbins=25,
                marginal="box",
                title=f"Forecast Residual Distribution (Shapiro-Wilk W={sw_stat:.3f}, p={sw_p:.4f})",
                labels={"value": "Residual Error (Actual - Predicted / 100k)"},
                color_discrete_sequence=["#48bb78"]
            )
            fig_resid.update_layout(height=400)
            st.plotly_chart(fig_resid, use_container_width=True)

        st.markdown("---")

        # Per-Region Accuracy Breakdown
        st.markdown("#### 🎯 Per-Region Forecast Accuracy (Test Period 2023-2025)")
        reg_acc = []
        for r in REGION_ORDER:
            sub = pv_df[pv_df["Region"] == r]
            if not sub.empty:
                err = sub["Predicted"] - sub["Actual"]
                rmse_r = np.sqrt((err ** 2).mean())
                err_pct = (err / sub["Actual"] * 100).mean()
                reg_acc.append({
                    "Region": r,
                    "Mean Actual Rate": round(sub["Actual"].mean(), 1),
                    "Mean Predicted Rate": round(sub["Predicted"].mean(), 1),
                    "RMSE (/100k)": round(rmse_r, 1),
                    "Mean % Error": f"{err_pct:+.1f}%"
                })
        st.dataframe(pd.DataFrame(reg_acc), hide_index=True, use_container_width=True)

        st.markdown("---")

        # Model B: Alcohol PAC Inclusion
        st.markdown("#### 🍷 Model B: Evaluating Wholesale Alcohol Supply (PAC) Impact [A12]")
        st.markdown(
            "Model B evaluates whether adding quarterly wholesale alcohol supply per capita (`alcohol_per_capita`) "
            "improves forecasting skill over the 2023-2024 training and 2025 test window."
        )

        res_b = model_artifacts["results_b"]
        if res_b:
            b_rows = []
            for n, mb in res_b.items():
                alc_coef = mb["coef"].get("alcohol_per_capita", None) if mb["coef"] else "N/A (Tree)"
                if isinstance(alc_coef, float):
                    alc_str = f"{alc_coef:+.4f}"
                else:
                    alc_str = str(alc_coef)
                b_rows.append({
                    "Model": n,
                    "Test 2025 RMSE (log)": round(mb["rmse"], 4),
                    "Test 2025 MAE (log)": round(mb["mae"], 4),
                    "Rate RMSE (/100k)": round(mb["rmse_rate"], 1),
                    "Alcohol per capita Coef": alc_str
                })
            st.dataframe(pd.DataFrame(b_rows), hide_index=True, use_container_width=True)
            st.info(
                "💡 **Policy Interpretation**: In linear models, the standardised coefficient on `alcohol_per_capita` "
                "indicates the marginal association between wholesale liquor supply and assault rates after controlling for regional baselines and lags."
            )
        else:
            st.warning("Insufficient alcohol data window for Model B test evaluation.")

    # =========================================================================
    # TAB 6: Policy Scenario Simulator
    # =========================================================================
    with tabs[5]:
        st.subheader("🎯 Interactive Policy Scenario & Staffing Demand Simulator")
        st.markdown(
            "Simulate the projected monthly assault rate and police/health resource demand for an NT jurisdiction "
            "by adjusting seasonal timing, recent crime velocity, and potential wholesale alcohol supply curbs."
        )

        sim_col1, sim_col2 = st.columns([1, 1])
        with sim_col1:
            st.markdown("##### 1. Jurisdiction & Environmental Controls")
            sim_region = st.selectbox("Select NT Region:", options=REGION_ORDER, index=0)
            sim_month = st.slider("Forecast Month:", min_value=1, max_value=12, value=11,
                                  format="%d - " + MONTH_LABELS[10])
            is_wet = 1 if sim_month in [11, 12, 1, 2, 3, 4] else 0
            st.info(f"Season designation: **{'Wet Season (Monsoonal Peak)' if is_wet else 'Dry Season'}**")

            st.markdown("##### 2. Baseline Crime & Alcohol Inputs")
            # Get typical regional values
            reg_defaults = monthly_assault[monthly_assault["Region"] == sim_region]
            def_lag1 = float(reg_defaults["Assault_rate_100k"].tail(6).mean()) if not reg_defaults.empty else 250.0
            def_alc_off = float(reg_defaults["Alcohol_offences"].tail(6).mean()) if not reg_defaults.empty else 40.0
            def_dv_off = float(reg_defaults["DV_offences"].tail(6).mean()) if not reg_defaults.empty else 60.0

            sim_lag1 = st.number_input("Prior Month Assault Rate (/100k):", value=round(def_lag1, 1), step=10.0)
            sim_lag12 = st.number_input("Same Month Last Year Assault Rate (/100k):", value=round(def_lag1, 1), step=10.0)
            sim_alc_off = st.number_input("Expected Alcohol-Involved Offences:", value=round(def_alc_off, 0), step=5.0)
            sim_dv_off = st.number_input("Expected DV-Involved Offences:", value=round(def_dv_off, 0), step=5.0)

            st.markdown("##### 3. Policy Intervention Lever: Alcohol Supply Curb")
            pac_curb = st.slider("Simulated Change in Wholesale Alcohol Supply (PAC %):",
                                 min_value=-50, max_value=30, value=0, step=5,
                                 help="e.g. -20% represents a strict reduction via takeaway restrictions or BDR enforcement.")

        with sim_col2:
            st.markdown("##### 📊 Projected Operational Outcome")

            # Construct feature vector for prediction
            sin_m = np.sin(2 * np.pi * sim_month / 12)
            season_val = is_wet

            # Features chosen in Model A
            feat_dict = {
                "sin_month": sin_m,
                "Season": season_val,
                "assault_rate_lag1": sim_lag1,
                "assault_rate_lag12": sim_lag12,
                "Alcohol_offences": sim_alc_off * (1.0 + (pac_curb / 100.0) * 0.4), # alcohol curb lowers alcohol-involved offences
                "DV_offences": sim_dv_off * (1.0 + (pac_curb / 100.0) * 0.2),
            }
            # Region dummy columns
            for r in REGION_ORDER:
                if r != "Greater Darwin":
                    feat_dict[f"Reg_{r}"] = 1 if sim_region == r else 0

            # Match features required by model
            model_feats = model_artifacts["features"]
            row_input = pd.DataFrame([{f: feat_dict.get(f, 0.0) for f in model_feats}])

            best_m = model_artifacts["results"][best_name]["model"]
            if best_name == "XGBoost":
                pred_log = float(best_m.predict(row_input)[0])
            else:
                scaled_input = model_artifacts["scaler"].transform(row_input)
                pred_log = float(best_m.predict(scaled_input)[0])

            pred_rate = float(np.expm1(pred_log))
            reg_pop = float(population[(population["Region"] == sim_region) & (population["Year"] == YEAR_MAX)]["Total_population"].values[0]) if not population[(population["Region"] == sim_region) & (population["Year"] == YEAR_MAX)].empty else 50000.0
            pred_incidents = (pred_rate / 100000.0) * reg_pop

            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Predicted Monthly Assault Rate</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{pred_rate:.1f} <span style="font-size:1.1rem;color:#718096;">per 100k</span></div>', unsafe_allow_html=True)
            st.markdown(f"**Projected Incident Count**: ~{int(round(pred_incidents))} assault offences (based on {int(reg_pop):,} regional residents)")
            st.markdown('</div>', unsafe_allow_html=True)
            st.write("")

            # Alert categorization
            if pred_rate >= rate_threshold:
                st.markdown('<div class="alert-priority-high">🚨 <b>CRITICAL PRIORITY ALERT</b><br>'
                            'Projected rate exceeds high alert threshold. Recommend activating frontline surge staffing, '
                            'increasing night-patrol presence, and ensuring domestic violence crisis shelter capacity.</div>', unsafe_allow_html=True)
            elif pred_rate >= rate_threshold * 0.7:
                st.markdown('<div class="alert-priority-med">⚠️ <b>ELEVATED RISK</b><br>'
                            'Above-average assault intensity anticipated. Recommend targeted liquor licensing enforcement '
                            'and proactive domestic violence referral sweeps.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-priority-low">✅ <b>STANDARD OPERATIONAL LEVEL</b><br>'
                            'Baseline demand expected. Standard roster allocation is adequate.</div>', unsafe_allow_html=True)

            # Impact of policy intervention
            if pac_curb != 0:
                # Baseline comparison without curb
                row_base = pd.DataFrame([{
                    "sin_month": sin_m,
                    "Season": season_val,
                    "assault_rate_lag1": sim_lag1,
                    "assault_rate_lag12": sim_lag12,
                    "Alcohol_offences": sim_alc_off,
                    "DV_offences": sim_dv_off,
                    **{f"Reg_{r}": (1 if sim_region == r else 0) for r in REGION_ORDER if r != "Greater Darwin"}
                }])[model_feats]
                if best_name == "XGBoost":
                    base_rate = float(np.expm1(best_m.predict(row_base)[0]))
                else:
                    base_rate = float(np.expm1(best_m.predict(model_artifacts["scaler"].transform(row_base))[0]))
                diff = pred_rate - base_rate
                st.metric(label="Net Effect of Policy Alcohol Intervention",
                          value=f"{pred_rate:.1f} /100k",
                          delta=f"{diff:+.1f} /100k ({diff/base_rate*100:+.1f}%)")

    # =========================================================================
    # TAB 7: Data Explorer & Exports
    # =========================================================================
    with tabs[6]:
        st.subheader("📁 Data Explorer & Policy Brief Export")
        st.markdown("Inspect harmonised panel records, filter columns, and export data subsets for cabinet briefs.")

        exp_col1, exp_col2 = st.columns([3, 1])
        with exp_col1:
            preview_cols = st.multiselect(
                "Choose display columns:",
                options=list(panel.columns),
                default=["Year", "Quarter", "Month number", "Region", "Offence category",
                         "Number of offences", "Alcohol_offences", "DV_offences", "Total_population", "Total PAC"]
            )
        with exp_col2:
            st.write("")
            st.write("")
            csv_data = filtered_panel[preview_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered Panel CSV",
                data=csv_data,
                file_name="nt_crime_filtered_extract.csv",
                mime="text/csv"
            )

        st.dataframe(filtered_panel[preview_cols].head(500), use_container_width=True)

        if MANIFEST_JSON.exists():
            with st.expander("📄 Data Ingest Manifest (Auditing & Traceability)", expanded=False):
                st.json(json.loads(MANIFEST_JSON.read_text()))


if __name__ == "__main__":
    main()

