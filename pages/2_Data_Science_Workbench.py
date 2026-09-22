"""
Data Science Workbench: NT Assault Forecasting & Econometric Modelling
=======================================================================
Tailored for Data Scientists, Econometricians, Machine Learning Engineers, and Peer Reviewers.
Exposes complete statistical tests, CV folds, VIF collinearity, PCA scree plots,
hyperparameter tuning curves, residual normality checks, paired t-tests, and Model A vs B comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st

from core_data import (
    ASSAULT_CATEGORY,
    EDA_PLOT_DIRS,
    MANIFEST_JSON,
    MONTH_LABELS,
    REG_PLOT_DIRS,
    REGION_COLORS,
    REGION_ORDER,
    YEAR_MAX,
    YEAR_MIN,
    get_regression_panel,
    load_data,
    run_model_training,
)

st.set_page_config(
    page_title="Data Science Workbench - NT Assault Forecasting",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .ds-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c5282;
        margin-bottom: 0.2rem;
    }
    .ds-sub {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .code-box {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="ds-header">🔬 Data Science Workbench & Econometric Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="ds-sub">Technical Validation, Cross-Validation Diagnostics, Feature Selection, and Residual Inference</div>', unsafe_allow_html=True)

# Load data & run models
panel, population, monthly_assault = load_data()
reg_panel, train, test, dummy_cols, cv_frame = get_regression_panel(monthly_assault, population)
model_artifacts = run_model_training(train, test, cv_frame, dummy_cols)

# -----------------------------------------------------------------------------
# Sidebar: Technical Controls
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Engineering & Modeling Parameters")
st.sidebar.markdown("Inspect training partitions, cross-validation protocols, and model hyperparameters.")

st.sidebar.markdown("**Cross-Validation Protocol:**")
st.sidebar.info("TimeSeriesSplit (k=5 folds) over calendar dates. StandardScaler refitted within each training fold to prevent distribution leakage [A15].")

st.sidebar.markdown("**Historical Partitions [A11]:**")
st.sidebar.markdown(f"- **Train Period**: {YEAR_MIN} – 2022 (`n={len(train)}` region-month rows)")
st.sidebar.markdown(f"- **Test Period**: 2023 – {YEAR_MAX} (`n={len(test)}` region-month rows)")
st.sidebar.markdown(f"- **Provisional Rows Flagged**: {reg_panel['is_provisional'].sum()} rows ([A9])")

# -----------------------------------------------------------------------------
# Technical Tabs
# -----------------------------------------------------------------------------
ds_tabs = st.tabs([
    "🏗️ Pipeline Architecture & Assumptions",
    "📊 Statistical EDA & Distributions",
    "👥 Age-Structure PCA [A14]",
    "🧪 Feature Selection & VIF [A16]",
    "⚙️ Hyperparameter Optimization",
    "🏆 Model Benchmarking (Model A & B)",
    "📉 Residual Diagnostics & Hypothesis Testing",
    "🖼️ High-Res Pipeline Plots"
])

# =============================================================================
# TAB 1: Architecture & Assumptions
# =============================================================================
with ds_tabs[0]:
    st.subheader("🏗️ Pipeline Data Engineering & Methodological Audit")
    st.markdown(
        "Complete provenance and assumption log implemented in `Updated_End_to_End_pipeline.py`. "
        "Every transformation is deterministic and preserves auditability."
    )

    col_a1, col_a2 = st.columns([3, 2])
    with col_a1:
        st.markdown("##### Pipeline Stage Execution Diagram")
        st.code("""
  [dataset/source/]
       │
       ▼ (Stage 1: Ingest & Schema Alignment)
  [dataset/raw/*.csv + manifest.json]
       │
       ▼ (Stage 2: Regional Harmonisation & Aggregation [A5, A8])
  [dataset/processed/nt_crime_merged_2015_2025.csv]  (13,629 rows x 44 cols)
       │
       ├───────────────────────────────┬───────────────────────────────┐
       ▼ (Stage 3: EDA & Correlation)  ▼ (Stage 4: PCA Demographics)  ▼ (Stage 5: Time-Series Regression)
  [eda_plots/*.png]              [Age-Structure Shares]         [Feature Variants -> VIF -> Tuning -> Final Models]
        """, language="text")

    with col_a2:
        if MANIFEST_JSON.exists():
            st.markdown("##### Raw Ingest Manifest (`manifest.json`)")
            manifest_dict = json.loads(MANIFEST_JSON.read_text())
            st.json(manifest_dict)
        else:
            st.info("Manifest file not found.")

    st.markdown("---")
    st.markdown("##### Assumption Log Reference Table [A1 – A16]")
    assumptions_data = [
        {"Code": "[A1]", "Name": "Historical Window", "Description": "Window fixed to 2015–2025; earlier records (from 2008) trimmed to avoid structural discontinuity."},
        {"Code": "[A2]", "Name": "ANZSOC Harmonisation", "Description": "Maps PROMIS offence categories to ANZSOC standards; unmapped categories throw an explicit error rather than silently becoming NaN."},
        {"Code": "[A3]", "Name": "Transition Gap (Nov 2023)", "Description": "November 2023 absent in both systems due to IT transition; maintained as visible NaN, not imputed with false values."},
        {"Code": "[A4]", "Name": "Unknown Region Resolution", "Description": "Reporting Region 'Unknown' mapped to residual 'Top End' region."},
        {"Code": "[A5]", "Name": "Population Region Alignment", "Description": "Harmonises police town data onto the 6 NTG population regions. Darwin and Palmerston merged into Greater Darwin to avoid duplicating population denominator."},
        {"Code": "[A6]", "Name": "Quarterly PAC Allocation", "Description": "Quarterly wholesale alcohol PAC attached across all 3 months of the respective quarter."},
        {"Code": "[A7]", "Name": "PAC Temporal Window", "Description": "PAC exists only for 2023–2025; earlier years remain NaN. Region x quarter mean fills only gaps within 2023–2025."},
        {"Code": "[A8]", "Name": "Offence Flag Accounting", "Description": "Alcohol and DV involvement counted in Number of Offences, not matching raw rows (schema changes split categories into more rows)."},
        {"Code": "[A9]", "Name": "Provisional Data", "Description": "Most recent 6 months flagged as provisional due to late-reporting lag."},
        {"Code": "[A10]", "Name": "Complete Calendar Grid", "Description": "Constructs complete Region x Month Cartesian grid so lag1, lag3, lag12 represent true calendar intervals across the Nov 2023 gap."},
        {"Code": "[A11]", "Name": "Model A Protocol", "Description": "Calendar + lag models without PAC; trained 2015–2022, tested 2023–2025."},
        {"Code": "[A12]", "Name": "Model B Protocol", "Description": "Evaluates marginal impact of PAC; trained 2023–2024, tested 2025. Different train window from Model A, hence errors are not directly comparable."},
        {"Code": "[A13]", "Name": "Calendar-Sorted CV", "Description": "TimeSeriesSplit applied over date-sorted rows to prevent training on one region and testing on another."},
        {"Code": "[A14]", "Name": "PCA on Age Shares", "Description": "PCA uses age-group shares (percentages), not counts, preventing regional population size from dominating PC1 at >99%."},
        {"Code": "[A15]", "Name": "Pipeline Scaler", "Description": "StandardScaler fitted inside Pipeline on training folds only, preventing lookahead distribution leakage."},
        {"Code": "[A16]", "Name": "Leakage-Free Selection", "Description": "Feature variants selected using TimeSeriesSplit CV on the training partition only; held-out test set remained untouched until final evaluation."},
    ]
    st.dataframe(pd.DataFrame(assumptions_data), hide_index=True, use_container_width=True)


# =============================================================================
# TAB 2: Statistical EDA & Distributions
# =============================================================================
with ds_tabs[1]:
    st.subheader("📊 Statistical EDA, Distribution Skewness, and Fixed Effects Correlation")

    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        # Skewness Analysis: Raw vs Log-transformed
        raw_assault = monthly_assault["Assault_offences"]
        log_assault = np.log1p(raw_assault)

        skew_fig = go.Figure()
        skew_fig.add_trace(go.Histogram(x=raw_assault, nbinsx=30, name=f"Raw (Skew={raw_assault.skew():.2f})", marker_color="#4299e1"))
        skew_fig.update_layout(title="Raw Assault Distribution (Positively Skewed)", height=320, xaxis_title="Assault Offences")
        st.plotly_chart(skew_fig, use_container_width=True)

    with eda_c2:
        skew_fig_log = go.Figure()
        skew_fig_log.add_trace(go.Histogram(x=log_assault, nbinsx=30, name=f"Log-Transformed (Skew={log_assault.skew():.2f})", marker_color="#48bb78"))
        skew_fig_log.update_layout(title="Log-Transformed log(1 + Assault) Distribution (Near Normal)", height=320, xaxis_title="log(1 + Assault Offences)")
        st.plotly_chart(skew_fig_log, use_container_width=True)

    st.markdown("---")

    # Econometric Correlation: Pooled vs Within-Region
    st.markdown("#### 🔬 Correlation Heatmap & Econometric Scale-Confounding Analysis")
    st.markdown(
        "When evaluating wholesale alcohol supply (Total PAC) against assault rates, **pooling all regions creates a severe scale confounder**: "
        "Larger regions consume more alcohol and record more crimes simply due to population scale. "
        "Below we report both the **pooled correlation** and the **within-region correlation**."
    )

    corr_col1, corr_col2 = st.columns([3, 2])
    with corr_col1:
        pac_sub = monthly_assault[monthly_assault["Total_PAC"].notna()].copy()
        corr_frame = pac_sub[["Assault_rate_100k", "alcohol_per_capita", "Total_population",
                              "Aboriginal", "Alcohol_offences", "DV_offences"]].rename(columns={
            "Assault_rate_100k": "Assault Rate /100k",
            "alcohol_per_capita": "PAC per Capita",
            "Total_population": "Total Population",
            "Aboriginal": "Aboriginal Pop",
            "Alcohol_offences": "Alcohol Offences",
            "DV_offences": "DV Offences"
        })
        corr_matrix = corr_frame.corr().round(2)
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Correlation Heatmap (Assault Predictors)")
        fig_corr.update_layout(height=380)
        st.plotly_chart(fig_corr, use_container_width=True)

    with corr_col2:
        pooled_r = pac_sub["alcohol_per_capita"].corr(pac_sub["Assault_rate_100k"])
        within = pac_sub.groupby("Region").apply(lambda g: g["alcohol_per_capita"].corr(g["Assault_rate_100k"])).round(3)

        st.markdown("##### Econometric Findings:")
        st.markdown(f"- **Pooled Pearson Correlation**: `r = {pooled_r:.3f}`")
        st.caption("Confounded by between-region population heterogeneity.")
        st.markdown("##### Within-Region (Fixed-Effects) Correlations:")
        st.dataframe(pd.DataFrame(within, columns=["Within-Region r"]), use_container_width=True)


# =============================================================================
# TAB 3: Demographic PCA [A14]
# =============================================================================
with ds_tabs[2]:
    st.subheader("👥 Principal Component Analysis on Regional Age Structure [A14]")
    st.markdown(
        "Under **[A14]**, PCA is fitted on regional **age-group shares (proportions)** rather than raw population headcounts. "
        "If raw counts were used, regional population size would account for >99% of PC1 variance, measuring only 'how large the region is' "
        "rather than demographic age composition."
    )

    age_cols = sorted(c for c in population.columns if c.startswith("Pop_age_"))
    shares = population[age_cols].div(population["Total_population"], axis=0)
    scaled_shares = StandardScaler().fit_transform(shares.astype(float).values)
    pca = PCA().fit(scaled_shares)
    explained = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)

    pca_col1, pca_col2 = st.columns([3, 2])
    with pca_col1:
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(explained))], y=explained, name="Individual Variance %", marker_color="#3182ce"))
        fig_scree.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(explained))], y=cumulative, name="Cumulative Variance %", marker_color="#e53e3e", mode="lines+markers"))
        fig_scree.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="90% Variance Threshold")
        fig_scree.update_layout(title="PCA Scree Plot: Regional Age Structure Variance", height=380, yaxis_title="Variance Explained (%)")
        st.plotly_chart(fig_scree, use_container_width=True)

    with pca_col2:
        pca_summary = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(explained))],
            "Explained %": explained.round(1),
            "Cumulative %": cumulative.round(1)
        })
        st.dataframe(pca_summary.head(8), hide_index=True, use_container_width=True)
        n_90 = int(np.searchsorted(cumulative, 90) + 1)
        st.success(f"**Dimensionality Reduction Result**: First **{n_90} principal components** capture >90% of total demographic age variance across the NT.")


# =============================================================================
# TAB 4: Feature Selection & VIF [A16]
# =============================================================================
with ds_tabs[3]:
    st.subheader("🧪 Leakage-Free Feature Selection & Multicollinearity Diagnostics [A16]")
    st.markdown(
        "In accordance with **[A16]**, feature variants were benchmarked strictly on the training partition "
        "(2015–2022) using 5-fold TimeSeriesSplit CV. The test partition (2023–2025) was withheld completely from model selection."
    )

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("##### R1: Feature Variant Comparison (TimeSeriesSplit CV)")
        v_rmse = model_artifacts["variant_rmse"]
        fig_var = px.bar(
            x=list(v_rmse.keys()),
            y=list(v_rmse.values()),
            color=list(v_rmse.keys()),
            color_discrete_sequence=["#90caf9", "#90caf9", "#90caf9", "#e53e3e"],
            title="Feature Variant CV RMSE (Lower is Better | Red = Winner)",
            labels={"x": "Feature Variant", "y": "CV RMSE (log scale)"}
        )
        fig_var.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_var, use_container_width=True)
        st.info(f"Selected Feature Variant: **{model_artifacts['best_variant']}**")

    with f_col2:
        st.markdown("##### R2: Multicollinearity & Variance Inflation Factor (VIF)")
        st.markdown("Features with VIF > 30 are eliminated. Features between 10–30 are monitored for stability.")
        st.dataframe(
            model_artifacts["vif_df"],
            column_config={
                "Feature": "Feature",
                "VIF": st.column_config.NumberColumn("VIF Score", format="%.1f"),
                "Status": "Collinearity Status"
            },
            hide_index=True,
            use_container_width=True
        )


# =============================================================================
# TAB 5: Hyperparameter Optimization
# =============================================================================
with ds_tabs[4]:
    st.subheader("⚙️ Hyperparameter Optimization Curves")
    st.markdown(
        "Tuning regularisation penalties (L2 Ridge $\alpha$, L1 Lasso $\alpha$) and tree architectures (XGBoost) "
        "via 5-fold TimeSeriesSplit over calendar order [A13]."
    )

    h_col1, h_col2 = st.columns(2)
    with h_col1:
        ridge_cv = model_artifacts["ridge_cv"]
        lasso_cv = model_artifacts["lasso_cv"]
        alphas = list(ridge_cv.keys())

        fig_alpha = go.Figure()
        fig_alpha.add_trace(go.Scatter(x=[str(a) for a in alphas], y=list(ridge_cv.values()), name="Ridge CV RMSE", mode="lines+markers", line=dict(color="#2196F3", width=2)))
        fig_alpha.add_trace(go.Scatter(x=[str(a) for a in alphas], y=list(lasso_cv.values()), name="Lasso CV RMSE", mode="lines+markers", line=dict(color="#FF9800", width=2)))
        fig_alpha.update_layout(title="Ridge & Lasso Alpha Tuning Curves", xaxis_title="Alpha (Regularization Penalty)", yaxis_title="5-Fold CV RMSE", height=360)
        st.plotly_chart(fig_alpha, use_container_width=True)
        st.caption(f"Optimal Ridge α: **{model_artifacts['best_ridge']}** | Optimal Lasso α: **{model_artifacts['best_lasso']}**")

    with h_col2:
        xgb_scores = model_artifacts["xgb_scores"]
        param_grid = model_artifacts["param_grid"]
        labels = [f"d={p['max_depth']}, lr={p['learning_rate']}, α={p['reg_alpha']}" for p in param_grid]

        fig_xgb = px.line(
            x=labels,
            y=list(xgb_scores.values()),
            markers=True,
            title="XGBoost Parameter Grid Tuning",
            labels={"x": "Architecture Parameters", "y": "5-Fold CV RMSE"}
        )
        fig_xgb.update_traces(line_color="#48bb78", line_width=2)
        fig_xgb.update_layout(height=360)
        st.plotly_chart(fig_xgb, use_container_width=True)
        st.caption(f"Best XGBoost Hyperparameters: `{model_artifacts['best_xgb_params']}`")


# =============================================================================
# TAB 6: Model Benchmarking (Model A & Model B)
# =============================================================================
with ds_tabs[5]:
    st.subheader("🏆 Model Benchmarking Leaderboard: Model A & Model B")

    res = model_artifacts["results"]
    best_name = model_artifacts["best_name"]

    st.markdown("#### Model A Evaluation (Held-Out Test: 2023–2025)")
    rows = []
    for name, m in res.items():
        rows.append({
            "Model Architecture": name,
            "Test RMSE (log)": round(m["rmse"], 4),
            "Test MAE (log)": round(m["mae"], 4),
            "Rate RMSE (/100k)": round(m["rmse_rate"], 1),
            "5-Fold CV RMSE": round(m["cv"], 4),
            "Verdict": "⭐ TOP MODEL" if name == best_name else "Benchmark"
        })
    st.dataframe(pd.DataFrame(rows).sort_values("Test RMSE (log)"), hide_index=True, use_container_width=True)

    st.markdown("---")

    # Standardized Coefficients & Feature Importance
    st.markdown("#### Standardized Coefficients (Ridge vs Lasso) & XGBoost Feature Importance")
    c_df1, c_df2 = st.columns(2)
    with c_df1:
        ridge_m = res[f"Ridge (α={model_artifacts['best_ridge']})"]["model"]
        lasso_m = res[f"Lasso (α={model_artifacts['best_lasso']})"]["model"]
        coef_df = pd.DataFrame({
            "Feature": model_artifacts["features"],
            "Ridge Coef": ridge_m.coef_.round(4),
            "Lasso Coef": lasso_m.coef_.round(4),
            "Lasso Selection": ["ZEROED OUT" if abs(c) < 1e-5 else "RETAINED" for c in lasso_m.coef_]
        }).sort_values("Ridge Coef", key=abs, ascending=False)
        st.dataframe(coef_df, hide_index=True, use_container_width=True)

    with c_df2:
        xgb_m = res["XGBoost"]["model"]
        imp_df = pd.DataFrame({
            "Feature": model_artifacts["features"],
            "XGBoost Importance": xgb_m.feature_importances_.round(4)
        }).sort_values("XGBoost Importance", ascending=False)
        fig_imp = px.bar(imp_df, x="XGBoost Importance", y="Feature", orientation="h",
                         title="XGBoost Relative Feature Importance", color="XGBoost Importance", color_continuous_scale="Blues")
        fig_imp.update_layout(height=360, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # Model B: Alcohol Supply Evaluation [A12]
    st.markdown("#### 🍷 Model B: Evaluating Wholesale Alcohol Supply (PAC) Marginal Skill [A12]")
    st.markdown(
        "Trained on 2023–2024 and evaluated on 2025 to measure whether adding quarterly PAC per capita "
        "improves out-of-sample forecast accuracy."
    )
    res_b = model_artifacts["results_b"]
    if res_b:
        b_rows = []
        for n, mb in res_b.items():
            alc_coef = mb["coef"].get("alcohol_per_capita", None) if mb["coef"] else "N/A (Tree)"
            alc_str = f"{alc_coef:+.4f}" if isinstance(alc_coef, float) else str(alc_coef)
            b_rows.append({
                "Model": n,
                "Test 2025 RMSE (log)": round(mb["rmse"], 4),
                "Test 2025 MAE (log)": round(mb["mae"], 4),
                "Rate RMSE (/100k)": round(mb["rmse_rate"], 1),
                "Standardised Alcohol Coef": alc_str
            })
        st.dataframe(pd.DataFrame(b_rows), hide_index=True, use_container_width=True)


# =============================================================================
# TAB 7: Residual Diagnostics & Hypothesis Testing
# =============================================================================
with ds_tabs[6]:
    st.subheader("📉 Residual Diagnostics & Hypothesis Testing")

    actual_rate = np.expm1(test["log_assault_rate"].to_numpy(float))
    pred_rate = np.expm1(res[best_name]["pred"])
    resid_log = test["log_assault_rate"].to_numpy(float) - res[best_name]["pred"]

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        # Predicted vs Actual Scatter
        pv_df = pd.DataFrame({
            "Actual": actual_rate,
            "Predicted": pred_rate,
            "Region": test["Region"].to_numpy()
        })
        fig_pva = px.scatter(pv_df, x="Actual", y="Predicted", color="Region", color_discrete_map=REGION_COLORS,
                             title=f"Predicted vs Actual ({best_name})", labels={"Actual": "Actual Rate (/100k)", "Predicted": "Predicted Rate (/100k)"})
        max_v = max(actual_rate.max(), pred_rate.max()) * 1.05
        fig_pva.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(color="red", dash="dash"))
        fig_pva.update_layout(height=380)
        st.plotly_chart(fig_pva, use_container_width=True)

    with r_col2:
        # Shapiro-Wilk Normality Test
        sw_stat, sw_p = stats.shapiro(resid_log)
        fig_res = px.histogram(resid_log, nbins=25, marginal="box",
                               title=f"Log Residual Distribution (Shapiro-Wilk W={sw_stat:.4f}, p={sw_p:.4f})",
                               labels={"value": "Residual log error"}, color_discrete_sequence=["#4299e1"])
        fig_res.update_layout(height=380)
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")

    # Paired t-tests
    st.markdown("#### Paired t-tests on Absolute Forecast Errors (α = 0.05)")
    st.markdown("Tests whether performance differences between model pairs are statistically significant.")
    names = list(res.keys())
    t_records = []
    y_test = test["log_assault_rate"].to_numpy(float)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            t_stat, p_val = stats.ttest_rel(np.abs(y_test - res[a]["pred"]), np.abs(y_test - res[b]["pred"]))
            t_records.append({
                "Model Comparison": f"{a} vs {b}",
                "t-statistic": round(t_stat, 4),
                "p-value": round(p_val, 4),
                "Null Hypothesis (H0)": "Reject H0 (Statistically Significant)" if p_val < 0.05 else "Fail to Reject H0 (No Sig. Difference)"
            })
    st.dataframe(pd.DataFrame(t_records), hide_index=True, use_container_width=True)


# =============================================================================
# TAB 8: High-Res Pipeline Plots Viewer
# =============================================================================
with ds_tabs[7]:
    st.subheader("🖼️ Original Pipeline High-Resolution Figure Viewer")
    st.markdown("Browse and inspect all pre-rendered figures generated by `Updated_End_to_End_pipeline.py`.")

    plot_category = st.radio("Figure Directory:", ["Exploratory Data Analysis (`eda_plots/`)", "Regression Diagnostics (`regression_plots/`)"], horizontal=True)

    target_dirs = EDA_PLOT_DIRS if "EDA" in plot_category else REG_PLOT_DIRS
    found_pngs = []
    for d in target_dirs:
        if d.exists():
            pngs = sorted(list(d.glob("*.png")))
            if pngs:
                found_pngs = pngs
                break

    if found_pngs:
        chosen_fig = st.selectbox("Select figure to render:", [f.name for f in found_pngs])
        fig_file = next(f for f in found_pngs if f.name == chosen_fig)
        st.image(str(fig_file), caption=f"Pipeline Figure: {chosen_fig}", use_container_width=True)
    else:
        st.info("No generated PNG figures found.")

