"""
Northern Territory Assault Forecasting & Crime Intelligence Portal
===================================================================
Main Portal Landing Page - Choose your operational role:
- 🏛️ Administrator Portal (Non-Technical / Policy Makers, Police, Health Planners)
- 🔬 Data Science Workbench (Technical / Econometrics, ML Engineers, Peer Reviewers)
"""

from __future__ import annotations

import streamlit as st
from core_data import load_data, YEAR_MAX, YEAR_MIN

st.set_page_config(
    page_title="NT Assault Forecasting & Crime Intelligence Portal",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .portal-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .portal-subtitle {
        font-size: 1.15rem;
        color: #4a5568;
        margin-bottom: 1.6rem;
    }
    .role-card {
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    .admin-card {
        background: linear-gradient(135deg, #f7fafc 0%, #ebf8ff 100%);
        border-left: 6px solid #3182ce;
    }
    .ds-card {
        background: linear-gradient(135deg, #f7fafc 0%, #f0fff4 100%);
        border-left: 6px solid #38a169;
    }
    .card-heading {
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .card-desc {
        font-size: 0.95rem;
        color: #4a5568;
        line-height: 1.5;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="portal-title">⚖️ Northern Territory Assault Forecasting & Crime Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="portal-subtitle">Dual-Role Decision Support System | PRT661 Data Science Practice (Group DAN2 Theme 2)</div>', unsafe_allow_html=True)

# Fast summary KPI banner
panel, population, monthly_assault = load_data()
total_offences = panel["Number of offences"].sum()
total_assaults = monthly_assault["Assault_offences"].sum()
avg_rate = monthly_assault["Assault_rate_100k"].mean()
latest_pop = population[population["Year"] == YEAR_MAX]["Total_population"].sum()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Recorded Offences", f"{total_offences:,.0f}")
with k2:
    st.metric("Recorded Assault Incidents", f"{total_assaults:,.0f}")
with k3:
    st.metric("Territory Avg Monthly Rate", f"{avg_rate:.1f} /100k")
with k4:
    st.metric("Territory Population (2025)", f"{latest_pop:,.0f}")

st.markdown("---")
st.subheader("🎯 Select Your Operational Workspace")
st.markdown("This portal provides two separated, dedicated interfaces tailored to distinct operational needs:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="role-card admin-card">
        <div class="card-heading" style="color: #2b6cb0;">🏛️ Administrator & Policy Portal</div>
        <div class="card-desc">
            Designed specifically for <b>Police Commissioners, Health Directors, Cabinet Advisors, and Frontline Resource Planners</b>.<br><br>
            <b>Key Features:</b>
            <ul>
                <li><b>Spatial Crime Distribution Map</b>: Interactive NT map with per-capita victimisation layers and service hubs.</li>
                <li><b>Frontline Staffing Matrix</b>: Critical, Elevated, and Baseline priority tiers with actionable operational responses.</li>
                <li><b>Seasonal Surge Calendar</b>: Wet Season vs Dry Season police roster and hospital ED planning.</li>
                <li><b>Policy Scenario Simulator</b>: Simulate real-world policy levers like wholesale alcohol supply curbs (-15%, -20%) and forecast incident reduction.</li>
                <li><b>Zero technical jargon</b>: Plain-English, high-impact executive summaries.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Enter Administrator Portal", type="primary", use_container_width=True):
        st.switch_page("pages/1_Administrator_Portal.py")

with col2:
    st.markdown("""
    <div class="role-card ds-card">
        <div class="card-heading" style="color: #276749;">🔬 Data Science Workbench</div>
        <div class="card-desc">
            Designed for <b>Data Scientists, Statisticians, Econometricians, and Peer Reviewers</b>.<br><br>
            <b>Key Features:</b>
            <ul>
                <li><b>Pipeline Provenance & Assumption Log [A1–A16]</b>: Audit raw ingest schemas, ANZSOC crosswalks, and transition gaps.</li>
                <li><b>Statistical EDA & Fixed Effects</b>: Skewness analysis, log transformations, pooled vs within-region econometric correlations.</li>
                <li><b>Demographic PCA Decomposition [A14]</b>: Age-group share scree plots and components explaining >90% variance.</li>
                <li><b>Leakage-Free Feature Selection & VIF [A16]</b>: TimeSeriesSplit CV and collinearity elimination.</li>
                <li><b>Model Benchmarking (Model A & B)</b>: Compare Linear Regression, Ridge, Lasso, and XGBoost with residual diagnostics and paired t-tests.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔬 Enter Data Science Workbench", use_container_width=True):
        st.switch_page("pages/2_Data_Science_Workbench.py")

st.markdown("---")
st.sidebar.markdown("### 🧭 Navigation")
st.sidebar.page_link("app.py", label="Home / Overview", icon="🏠")
st.sidebar.page_link("pages/1_Administrator_Portal.py", label="Administrator Portal", icon="🏛️")
st.sidebar.page_link("pages/2_Data_Science_Workbench.py", label="Data Science Workbench", icon="🔬")

st.markdown("##### 📌 Methodology Summary")
st.markdown("""
This project forecasts assault offences across the 6 official Northern Territory administrative regions 
(*Greater Darwin, Central Australia, Big Rivers, East Arnhem, Barkly, Top End*) from 2015 to 2025.
By linking monthly crime statistics with annual ABS demographic tables and quarterly wholesale alcohol supply (Pure Alcohol Content),
the system delivers both operational resource planning recommendations and econometric policy evaluations.
""")
