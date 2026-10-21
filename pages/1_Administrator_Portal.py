"""
Administrator Dashboard: Northern Territory Crime & Resource Planning Portal
=============================================================================
Tailored for Policy Makers, Police Commissioners, Health Directors, and Justice Planners.
Focuses on operational decision-making, resource allocation, spatial hot spots, and policy levers.
Zero complex statistical jargon - pure actionable intelligence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core_data import (
    KEY_TOWNS,
    MONTH_LABELS,
    REGION_COLORS,
    REGION_GEO,
    REGION_ORDER,
    YEAR_MAX,
    YEAR_MIN,
    get_regression_panel,
    latest_population,
    load_data,
    load_forecast_outputs,
    load_saved_model_a,
    run_model_training,
)

st.set_page_config(
    page_title="Administrator Portal - NT Crime & Resource Planning",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Executive Government Look
st.markdown("""
<style>
    .admin-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .admin-sub {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border: 1px solid #cbd5e0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        color: #1a202c;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #718096;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .alert-critical {
        background-color: #fff5f5;
        border-left: 5px solid #e53e3e;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        color: #742a2a;
    }
    .alert-elevated {
        background-color: #fffaf0;
        border-left: 5px solid #dd6b20;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        color: #7b341e;
    }
    .alert-baseline {
        background-color: #f0fff4;
        border-left: 5px solid #38a169;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        color: #22543d;
    }
    /* Dark mode (Streamlit "System" theme uses prefers-color-scheme) */
    @media (prefers-color-scheme: dark) {
        .admin-header {
            color: #90cdf4;
        }
        .admin-sub {
            color: #a0aec0;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="admin-header">🏛️ Northern Territory Justice & Public Safety Resource Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="admin-sub">Executive Decision Support for Frontline Policing, Health Services, and Policy Planning</div>', unsafe_allow_html=True)

# Load data. Forecasts come from Outputs/; Model A is loaded in Tab 6 from
# Models/model_a.joblib. Do not refit on page load.
panel, population, monthly_assault = load_data()

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Operational Parameters")
st.sidebar.markdown("Filter jurisdictions and historical calendar windows for executive review.")

selected_regions = st.sidebar.multiselect(
    "Target Administrative Regions",
    options=REGION_ORDER,
    default=REGION_ORDER,
    help="Select one or more NT Government regions."
)
if not selected_regions:
    selected_regions = REGION_ORDER

year_range = st.sidebar.slider(
    "Review Period (Years)",
    min_value=YEAR_MIN,
    max_value=YEAR_MAX,
    value=(YEAR_MIN, YEAR_MAX),
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Priority Thresholds")
rate_threshold = st.sidebar.slider(
    "High-Need Rate Threshold (/100k / month)",
    min_value=150, max_value=700, value=350, step=25,
    help="Per-capita rate above which a region is flagged for additional service capacity."
)

# Filtered data
filtered_panel = panel[
    panel["Region"].isin(selected_regions) &
    panel["Year"].between(year_range[0], year_range[1])
]
filtered_assault = monthly_assault[
    monthly_assault["Region"].isin(selected_regions) &
    monthly_assault["Year"].between(year_range[0], year_range[1])
]

# -----------------------------------------------------------------------------
# Executive Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📊 Executive Briefing & KPIs",
    "📈 Forecasts & Planning Ranges",
    "🗺️ Crime Map & Spatial Resource Demand",
    "🚨 Resource Priority & Staffing Matrix",
    "📅 Seasonal Demand Calendar",
    "🎯 Policy Intervention Simulator",
    "📥 Briefing Note Export"
])

# =============================================================================
# TAB 1: Executive Briefing & KPIs
# =============================================================================
with tabs[0]:
    st.subheader(f"Territory-Wide Crime Indicators ({year_range[0]} - {year_range[1]})")

    total_offences = filtered_panel["Number of offences"].sum()
    total_assaults = filtered_assault["Assault_offences"].sum()
    avg_assault_rate = filtered_assault["Assault_rate_100k"].mean()
    alc_assault_share = (filtered_assault["Alcohol_offences"].sum() / total_assaults * 100) if total_assaults > 0 else 0
    dv_assault_share = (filtered_assault["DV_offences"].sum() / total_assaults * 100) if total_assaults > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Recorded Offences", f"{total_offences:,.0f}")
    with col2:
        st.metric("Total Assault Incidents", f"{total_assaults:,.0f}")
    with col3:
        st.metric("Avg Monthly Assault Rate", f"{avg_assault_rate:.1f} /100k")
    with col4:
        st.metric("Alcohol-Involved Share", f"{alc_assault_share:.1f}%")
    with col5:
        st.metric("Domestic Violence Share", f"{dv_assault_share:.1f}%")

    st.markdown("---")

    # High-level summary narrative
    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.markdown("#### 📌 Key Findings")
        st.markdown(f"""
        - **Assault is the second most recorded offence**: across 2015-2025 only theft was recorded more often, and assault is the offence this system forecasts.
        - **High Alcohol & Domestic Violence Nexus**: **{alc_assault_share:.1f}%** of all recorded assaults are alcohol-involved, and **{dv_assault_share:.1f}%** involve domestic or family violence. Wholesale alcohol supply is positively associated with assault rates, but it did not improve the forecasts and its causal effect is not established (report, item 5.3).
        - **Per-Capita Disparity**: Greater Darwin generates the largest absolute headcount of incidents due to its population size (about 60% of NT residents), but remote regional jurisdictions (such as Central Australia and Barkly) suffer from significantly higher per-capita assault rates.
        """)

    with c_right:
        st.markdown("#### ⚡ Frontline Alert Summary")
        latest_year = year_range[1]
        latest_rates = filtered_assault[filtered_assault["Year"] == latest_year].groupby("Region")["Assault_rate_100k"].mean()
        critical_regs = latest_rates[latest_rates >= rate_threshold].index.tolist()
        elevated_regs = latest_rates[(latest_rates < rate_threshold) & (latest_rates >= rate_threshold * 0.7)].index.tolist()

        if critical_regs:
            st.markdown(f'<div class="alert-critical">🔴 <b>HIGH RESOURCE NEED</b><br>'
                        f'Regions above the planning threshold ({rate_threshold}/100k): <b>{", ".join(critical_regs)}</b>.<br>'
                        f'Candidates for a capacity review at the next planning cycle, alongside local operational knowledge.</div>', unsafe_allow_html=True)
        if elevated_regs:
            st.markdown(f'<div class="alert-elevated">⚠️ <b>ELEVATED MONITORING</b><br>'
                        f'Regions in elevated tier: <b>{", ".join(elevated_regs)}</b>.<br>'
                        f'Watch at the next data release; compare with the forecast ranges.</div>', unsafe_allow_html=True)
        if not critical_regs and not elevated_regs:
            st.markdown('<div class="alert-baseline">✅ <b>BASELINE OPERATIONAL TIER</b><br>All selected jurisdictions are operating within standard historical limits.</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2: Forecasts & Planning Ranges (Assessment 4)
# =============================================================================
# Reads Outputs/ written by evaluation.py, forecast.py and monitoring.py, so
# the numbers here are the ones in the final report. Nothing is refitted.
with tabs[1]:
    st.subheader("📈 Assault Forecasts and Planning Ranges")
    fo = load_forecast_outputs()
    fc_all = fo["forecasts"]
    mon = fo["monitoring"]

    if fc_all is None:
        st.info("Forecasts not found. Run `python forecast.py` after the pipeline and "
                "`evaluation.py` to create Outputs/F1_forecasts.csv.")
    else:
        rec = fc_all[fc_all["Recommended"].astype(str) == "True"].copy()
        origin = pd.to_datetime(rec["Origin"].iloc[0])
        method_by_h = rec.groupby("Horizon_months")["Model"].first().to_dict()

        # Release check banner (monitoring.py)
        if mon is not None:
            css = {"PASS": "alert-baseline", "WARN": "alert-elevated",
                   "FAIL": "alert-critical"}[mon["overall_status"]]
            rel = mon.get("forecast_release") or {}
            if mon["promote_forecasts_to_dashboard"]:
                verdict = "Cleared for planning use; warnings are reviewed at the next release."
            elif rel.get("action") == "rejected":
                verdict = (f"A release check failed, so the new forecasts were not published. The ranges below "
                           f"are the last approved forecasts (origin {rel.get('approved_origin', 'unknown')}).")
            else:
                verdict = "A release check failed: do not use these forecasts in briefings until reviewed."
            st.markdown(
                f'<div class="{css}"><b>Release check: {mon["overall_status"]}</b> '
                f'({mon["counts"]["PASS"]} pass, {mon["counts"]["WARN"]} warning, '
                f'{mon["counts"]["FAIL"]} fail). Crime data as at {mon["crime_data_as_at"]}, '
                f'checked {mon["check_date"]}.<br>{verdict}</div>',
                unsafe_allow_html=True)
        else:
            st.caption("Release checks have not been run yet (`python monitoring.py`).")

        st.markdown(
            f"Forecasts made at the end of **{origin:%B %Y}** for each region. Each horizon uses the "
            f"method with the lowest error in the rolling-origin backtest: "
            + "; ".join(f"**{h} month{'s' if h > 1 else ''}**: {m}" for h, m in method_by_h.items())
            + ". The range is an 80% planning range: about four months in five should fall inside it."
        )

        pop = latest_population(monthly_assault)
        rec = rec[rec["Region"].isin(selected_regions)].copy()
        rec["Population"] = rec["Region"].map(pop)
        for col, out in [("Forecast_rate_100k", "Expected"), ("Lower_80", "Low"), ("Upper_80", "High")]:
            rec[out] = (rec[col] * rec["Population"] / 100_000).round(0)
        rec["Target"] = pd.to_datetime(rec["Target_month"]).dt.strftime("%b %Y")
        rec["Rate range"] = rec.apply(lambda r: f"{r['Lower_80']:.0f} to {r['Upper_80']:.0f}", axis=1)
        rec["Assault range"] = rec.apply(lambda r: f"{r['Low']:.0f} to {r['High']:.0f}", axis=1)
        rec["Region"] = pd.Categorical(rec["Region"], REGION_ORDER, ordered=True)
        table = rec.sort_values(["Region", "Horizon_months"])[
            ["Region", "Target", "Horizon_months", "Model", "Forecast_rate_100k",
             "Rate range", "Expected", "Assault range"]]
        st.dataframe(
            table,
            column_config={
                "Target": "Month",
                "Horizon_months": st.column_config.NumberColumn("Months ahead", format="%d"),
                "Model": "Method",
                "Forecast_rate_100k": st.column_config.NumberColumn("Forecast (/100k)", format="%.0f"),
                "Rate range": "80% range (/100k)",
                "Expected": st.column_config.NumberColumn("Expected assaults", format="%.0f"),
                "Assault range": "80% range (assaults)",
            },
            hide_index=True, use_container_width=True,
        )
        st.caption("Expected assaults = forecast rate x the region's latest published population "
                   "(2025). Forecasts are region-level planning ranges, not predictions about "
                   "communities, places or individuals.")

        # Chart for one region
        chart_region = st.selectbox("Show forecast chart for", options=[r for r in REGION_ORDER if r in selected_regions])
        hist = monthly_assault[monthly_assault["Region"] == chart_region].copy()
        hist["date"] = pd.to_datetime(dict(year=hist["Year"], month=hist["Month number"], day=1))
        hist = hist[hist["date"] > origin - pd.DateOffset(months=36)].sort_values("date")
        reg_fc = rec[rec["Region"] == chart_region].copy()
        reg_fc["date"] = pd.to_datetime(reg_fc["Target_month"])

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=hist["date"], y=hist["Assault_rate_100k"], mode="lines",
                                    name="Observed", line=dict(color="#2d3748", width=2)))
        newest = fo["holdout_predictions"]
        if newest is not None:
            new_actual = (newest[newest["Region"] == chart_region]
                          .drop_duplicates("target")[["target", "actual"]])
            fig_fc.add_trace(go.Scatter(
                x=pd.to_datetime(new_actual["target"]), y=new_actual["actual"], mode="markers",
                name="2026 provisional actuals (not used by the models)",
                marker=dict(symbol="circle-open", size=9, color="#2d3748")))
        method_colours = {"XGBoost": "#E53935", "Moving average (12m)": "#1E88E5",
                          "Seasonal naive (t-12)": "#546E7A"}
        for model, grp in reg_fc.groupby("Model"):
            colour = method_colours.get(model, "#8E24AA")
            fig_fc.add_trace(go.Scatter(
                x=grp["date"], y=grp["Forecast_rate_100k"], mode="markers",
                name=f"Forecast: {model}", marker=dict(size=10, color=colour),
                error_y=dict(type="data", symmetric=False,
                             array=grp["Upper_80"] - grp["Forecast_rate_100k"],
                             arrayminus=grp["Forecast_rate_100k"] - grp["Lower_80"],
                             color=colour, thickness=2, width=6)))
        fig_fc.add_vline(x=origin, line_dash="dash", line_color="#a0aec0")
        fig_fc.update_layout(height=460, yaxis_title="Assaults per 100,000",
                             title=f"{chart_region}: observed rate and forecasts with 80% range",
                             xaxis=dict(tickformat="%b %Y"),
                             margin=dict(t=60, b=110),
                             legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
        st.plotly_chart(fig_fc, use_container_width=True)

        with st.expander("How reliable are these forecasts?"):
            bt = fo["backtest"]
            pi = fo["intervals"]
            ho = fo["holdout_metrics"]
            lines = []
            if bt is not None:
                for h, m in method_by_h.items():
                    row = bt[(bt["Model"] == m) & (bt["Horizon_months"] == h)]
                    if not row.empty:
                        r = row.iloc[0]
                        gain = (f"{r['Skill_%']:.0f}% more accurate than repeating last year's figure."
                                if m != "Seasonal naive (t-12)" else
                                "this is last year's figure for the same month; no fitted model beat it this far ahead.")
                        lines.append(f"- **{h} month{'s' if h > 1 else ''} ahead ({m})**: typical error "
                                     f"{r['MAE']:.0f} per 100,000 (RMSE {r['RMSE']:.0f}) over {int(r['n'])} past "
                                     f"forecasts; {gain}")
            if pi is not None:
                cov = ", ".join(f"{p['Out_of_sample_coverage_%']:.0f}% at {int(p['Horizon_months'])} m"
                                for _, p in pi.iterrows())
                lines.append(f"- **Planning ranges** held the outcome {cov} of the time (target 80%). "
                             "They are pooled across regions: wider than needed for Greater Darwin, "
                             "narrower than needed for Barkly and Top End.")
            if ho is not None:
                h1 = ho[(ho["Horizon_months"] == 1) & (ho["Recommended"].astype(str) == "True")]
                if not h1.empty:
                    r = h1.iloc[0]
                    lines.append(f"- **First test on 2026 data** ({r['Targets']}, provisional): "
                                 f"one-month error {r['RMSE']:.0f} against {r['Backtest_RMSE']:.0f} in the backtest.")
            st.markdown("\n".join(lines))

        if fo["checks"] is not None:
            with st.expander("Release check details (monitoring.py)"):
                st.dataframe(fo["checks"][["Status", "Check_id", "Check", "Value", "Note"]],
                             hide_index=True, use_container_width=True)

# =============================================================================
# TAB 3: Spatial Crime Map & Resource Demand
# =============================================================================
with tabs[2]:
    st.subheader("🗺️ Northern Territory Spatial Crime Distribution")
    st.markdown(
        "Visualise crime density across the Territory. "
        "Toggle between **Per-Capita Assault Rate** (how common assault is relative to population) "
        "and **Total Incident Headcount** (where the most incidents are recorded)."
    )

    m_col1, m_col2 = st.columns([1, 3])
    with m_col1:
        map_metric = st.radio(
            "Select Map Display Metric:",
            options=[
                "Per-Capita Assault Rate (/100k)",
                "Total Assault Headcount",
                "Alcohol-Involved Offences",
                "Domestic Violence Offences"
            ]
        )

        map_year = st.selectbox(
            "Select Calendar Year:",
            options=sorted(filtered_assault["Year"].unique(), reverse=True),
            index=0
        )

        show_towns = st.checkbox("Overlay Police & Hospital Service Hubs", value=True)

    # Prepare map points
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

    if map_metric.startswith("Per-Capita"):
        size_col = "Avg_Monthly_Rate"
        color_col = "Avg_Monthly_Rate"
        color_scale = "Reds"
        legend_title = "Monthly Rate / 100k"
    elif map_metric.startswith("Total Assault"):
        size_col = "Assault_offences"
        color_col = "Assault_offences"
        color_scale = "Viridis"
        legend_title = "Annual Assaults"
    elif map_metric.startswith("Alcohol-Involved"):
        size_col = "Alcohol_offences"
        color_col = "Alcohol_offences"
        color_scale = "Oranges"
        legend_title = "Alcohol Offences"
    else:
        size_col = "DV_offences"
        color_col = "DV_offences"
        color_scale = "Purples"
        legend_title = "DV Offences"

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
            title=f"NT Crime Distribution ({map_year}) - {legend_title}"
        )

        fig_map.update_geos(
            center=dict(lat=-18.0, lon=133.5),
            lataxis_range=[-26.5, -10.5],
            lonaxis_range=[128.5, 138.5],
            visible=True,
            showcoastlines=True,
            showland=True,
            landcolor="#f4f6f8",
            oceancolor="#e9f2f9",
            showocean=True,
            showrivers=True,
            showlakes=True
        )
        fig_map.update_layout(height=520, margin=dict(r=10, t=40, b=10, l=10))

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

    # The Resource Allocation Discrepancy
    st.markdown("---")
    st.markdown("#### ⚖️ The Resource Planning Discrepancy: Volume vs Per-Capita Burden")
    st.markdown(
        "**Volume and rate tell different stories**: Greater Darwin records the most assaults because it has the most "
        "residents, while Central Australia and Barkly have much higher rates per 100,000."
    )

    c_bar1, c_bar2 = st.columns(2)
    with c_bar1:
        fig_vol = px.bar(
            map_df_year.sort_values("Assault_offences", ascending=False),
            x="Region",
            y="Assault_offences",
            color="Region",
            color_discrete_map=REGION_COLORS,
            title=f"Total Assault Headcount ({map_year})",
            labels={"Assault_offences": "Recorded Assaults"}
        )
        fig_vol.update_layout(showlegend=False, height=330)
        st.plotly_chart(fig_vol, use_container_width=True)

    with c_bar2:
        fig_rate = px.bar(
            map_df_year.sort_values("Avg_Monthly_Rate", ascending=False),
            x="Region",
            y="Avg_Monthly_Rate",
            color="Region",
            color_discrete_map=REGION_COLORS,
            title=f"Per-Capita Assault Rate per 100,000 Residents ({map_year})",
            labels={"Avg_Monthly_Rate": "Monthly Rate / 100k"}
        )
        fig_rate.add_hline(y=rate_threshold, line_dash="dash", line_color="red",
                           annotation_text=f"Alert Threshold ({rate_threshold})")
        fig_rate.update_layout(showlegend=False, height=330)
        st.plotly_chart(fig_rate, use_container_width=True)


# =============================================================================
# TAB 4: Resource Priority & Staffing Matrix
# =============================================================================
with tabs[3]:
    st.subheader("🚨 Frontline Staffing & Operational Allocation Matrix")
    st.markdown(
        "This matrix ranks regions by their historical assault rate to support planning discussions. "
        "The tiers describe recorded demand against the threshold set in the sidebar; they are not "
        "validated resource prescriptions, and decisions should combine them with the forecasts and "
        "local operational knowledge."
    )

    reg_summary = filtered_assault.groupby("Region").agg(
        Total_Assaults=("Assault_offences", "sum"),
        Avg_Monthly_Rate=("Assault_rate_100k", "mean"),
        Alcohol_Share=("Alcohol_offences", lambda x: (x.sum() / filtered_assault.loc[x.index, "Assault_offences"].sum() * 100) if filtered_assault.loc[x.index, "Assault_offences"].sum() > 0 else 0),
        DV_Share=("DV_offences", lambda x: (x.sum() / filtered_assault.loc[x.index, "Assault_offences"].sum() * 100) if filtered_assault.loc[x.index, "Assault_offences"].sum() > 0 else 0),
    ).reset_index()

    def determine_action(row):
        rate = row["Avg_Monthly_Rate"]
        if rate >= rate_threshold:
            return "Above the planning threshold: review capacity with regional managers."
        elif rate >= rate_threshold * 0.7:
            return "Within 30% of the threshold: watch the next release and the forecast range."
        else:
            return "Below the threshold: no change indicated by recorded demand."

    reg_summary["Alert Status"] = reg_summary["Avg_Monthly_Rate"].apply(
        lambda r: "🔴 HIGH NEED" if r >= rate_threshold else ("🟠 ELEVATED" if r >= rate_threshold * 0.7 else "🟢 BASELINE")
    )
    reg_summary["Planning note"] = reg_summary.apply(determine_action, axis=1)
    reg_summary["Avg_Monthly_Rate"] = reg_summary["Avg_Monthly_Rate"].round(1)
    reg_summary["Alcohol_Share"] = reg_summary["Alcohol_Share"].round(1).astype(str) + "%"
    reg_summary["DV_Share"] = reg_summary["DV_Share"].round(1).astype(str) + "%"

    st.dataframe(
        reg_summary.sort_values("Avg_Monthly_Rate", ascending=False),
        column_config={
            "Region": "Jurisdiction",
            "Alert Status": "Priority Tier",
            "Avg_Monthly_Rate": st.column_config.NumberColumn("Avg Rate (/100k/mo)", format="%.1f"),
            "Total_Assaults": st.column_config.NumberColumn("Total Assaults", format="%d"),
            "Alcohol_Share": "Alcohol %",
            "DV_Share": "DV %",
            "Planning note": "Planning note (descriptive)"
        },
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# TAB 5: Seasonal Demand Calendar
# =============================================================================
with tabs[4]:
    st.subheader("📅 Northern Territory Seasonal Demand Calendar")
    st.markdown(
        "Recorded assaults vary across the year. The chart compares the **wet season (November to April)** "
        "with the **dry season (May to October)**; it describes the pattern and does not explain its causes."
    )

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        # Monthly Seasonality
        monthly_season = filtered_assault.groupby("Month number", as_index=False)["Assault_offences"].sum()
        monthly_season["Month"] = monthly_season["Month number"].map(lambda m: MONTH_LABELS[m - 1])
        monthly_season["Season"] = monthly_season["Month number"].apply(lambda m: "Wet Season" if m in [11, 12, 1, 2, 3, 4] else "Dry Season")

        fig_season = px.bar(
            monthly_season,
            x="Month",
            y="Assault_offences",
            color="Season",
            color_discrete_map={"Wet Season": "#EF5350", "Dry Season": "#42A5F5"},
            title="Total Assault Volume by Month (Wet vs Dry Season)",
            labels={"Assault_offences": "Total Assaults"}
        )
        fig_season.update_layout(height=380)
        st.plotly_chart(fig_season, use_container_width=True)

    with s_col2:
        st.markdown("##### 📋 Seasonal Pattern in the Selected Period")
        wet_mean = monthly_season.loc[monthly_season["Month number"].isin([11, 12, 1, 2, 3, 4]), "Assault_offences"].mean()
        dry_mean = monthly_season.loc[~monthly_season["Month number"].isin([11, 12, 1, 2, 3, 4]), "Assault_offences"].mean()
        top3 = monthly_season.sort_values("Assault_offences", ascending=False)["Month"].head(3).tolist()
        diff_pct = (wet_mean / dry_mean - 1) * 100 if dry_mean else 0.0
        st.markdown(f"""
        - **Wet season (Nov to Apr)**: on average **{wet_mean:,.0f}** recorded assaults per calendar month across the selected years.
        - **Dry season (May to Oct)**: on average **{dry_mean:,.0f}**, so the wet season is **{diff_pct:+.0f}%** different.
        - **Highest months**: {", ".join(top3)}.
        """)
        st.caption("Descriptive pattern from recorded data. Leave and roster timing should be planned with the "
                   "forecasts and their 80% ranges (Forecasts & Planning Ranges tab), not from this chart alone.")

    # Year x Month Matrix
    st.markdown("---")
    st.markdown("##### Historical Intensity Matrix (Year × Month)")
    heat = filtered_assault.groupby(["Year", "Month number"])["Assault_offences"].sum().unstack()
    heat.columns = [MONTH_LABELS[m - 1] for m in heat.columns]
    
    fig_heat = px.imshow(
        heat,
        labels=dict(x="Month", y="Year", color="Assaults"),
        x=list(heat.columns),
        y=list(heat.index),
        color_continuous_scale="YlOrRd",
        title="Year x Month Historical Heatmap (Note: Nov 2023 was systems transition gap)"
    )
    fig_heat.update_layout(height=360)
    st.plotly_chart(fig_heat, use_container_width=True)


# =============================================================================
# TAB 6: Policy Intervention Simulator
# =============================================================================
with tabs[5]:
    st.subheader("🎯 Policy Intervention & Scenario Simulator")
    st.markdown(
        "Explore an **illustrative sensitivity scenario**: how the model's projection changes when wholesale "
        "alcohol supply is assumed to change. It does not estimate the effect of any real policy."
    )

    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        st.markdown("##### 1. Jurisdiction & Environmental Context")
        sim_region = st.selectbox("Target Jurisdiction:", options=REGION_ORDER, index=0)
        sim_month = st.slider("Target Forecast Month:", min_value=1, max_value=12, value=12)
        is_wet = 1 if sim_month in [11, 12, 1, 2, 3, 4] else 0
        st.caption(
            f"Month: **{MONTH_LABELS[sim_month - 1]}** · "
            f"Climate Period: **{'Wet Season (Monsoonal Peak)' if is_wet else 'Dry Season'}**"
        )

        reg_defaults = monthly_assault[monthly_assault["Region"] == sim_region]
        def_lag1 = float(reg_defaults["Assault_rate_100k"].tail(6).mean()) if not reg_defaults.empty else 250.0
        def_alc_off = float(reg_defaults["Alcohol_offences"].tail(6).mean()) if not reg_defaults.empty else 40.0
        def_dv_off = float(reg_defaults["DV_offences"].tail(6).mean()) if not reg_defaults.empty else 60.0

        st.markdown("##### 2. Policy Lever: Alcohol Supply Intervention")
        pac_curb = st.slider(
            "Simulated Reduction / Surge in Wholesale Alcohol Supply (%):",
            min_value=-50, max_value=25, value=-15, step=5,
            help="e.g. -15% simulates stricter trading hours or liquor takeaway restrictions."
        )

        st.markdown("##### 3. Recent Jurisdictional Baseline")
        sim_lag1 = st.number_input("Prior Month Assault Rate (/100k):", value=round(def_lag1, 1), step=10.0)

    with sim_col2:
        st.markdown("##### 📊 Illustrative Sensitivity Scenario")

        # Feature vector. The models use the PREVIOUS month's alcohol- and
        # DV-flagged assault counts (pipeline [A18]); the simulator scales those.
        sin_m = np.sin(2 * np.pi * sim_month / 12)
        cos_m = np.cos(2 * np.pi * sim_month / 12)
        # Use the saved, corrected Model A bundle ([A21]); fall back to the
        # in-session fit only if Models/model_a.joblib is missing.
        saved_a = load_saved_model_a()
        if saved_a is not None:
            model_feats, best_name = saved_a["features"], saved_a["best_name"]
            best_m, sim_scaler, sim_period = saved_a["model"], saved_a["scaler"], saved_a["train_period"]
        else:
            # Fallback only when the saved bundle is missing.
            _, train, test, dummy_cols, cv_frame = get_regression_panel(monthly_assault, population)
            artifacts = run_model_training(train, test, cv_frame, dummy_cols)
            model_feats = artifacts["features"]
            best_name = artifacts["best_name"]
            best_m = artifacts["results"][best_name]["model"]
            sim_scaler, sim_period = artifacts["scaler"], "2016-2022"

        # Predict with simulated policy curb
        feat_dict = {
            "sin_month": sin_m,
            "cos_month": cos_m,
            "Season": is_wet,
            "assault_rate_lag1": sim_lag1,
            "assault_rate_lag3": sim_lag1,
            "assault_rate_lag12": sim_lag1,
            "Alcohol_offences_lag1": def_alc_off * (1.0 + (pac_curb / 100.0) * 0.4),
            "DV_offences_lag1": def_dv_off * (1.0 + (pac_curb / 100.0) * 0.2),
        }
        for r in REGION_ORDER:
            if r != "Greater Darwin":
                feat_dict[f"Reg_{r}"] = 1 if sim_region == r else 0

        row_input = pd.DataFrame([{f: feat_dict.get(f, 0.0) for f in model_feats}])
        if best_name == "XGBoost":
            pred_log = float(best_m.predict(row_input)[0])
        else:
            pred_log = float(best_m.predict(sim_scaler.transform(row_input))[0])
        pred_rate = float(np.expm1(pred_log))

        # Predict baseline without policy curb
        feat_dict_base = {
            "sin_month": sin_m,
            "cos_month": cos_m,
            "Season": is_wet,
            "assault_rate_lag1": sim_lag1,
            "assault_rate_lag3": sim_lag1,
            "assault_rate_lag12": sim_lag1,
            "Alcohol_offences_lag1": def_alc_off,
            "DV_offences_lag1": def_dv_off,
        }
        for r in REGION_ORDER:
            if r != "Greater Darwin":
                feat_dict_base[f"Reg_{r}"] = 1 if sim_region == r else 0

        row_base = pd.DataFrame([{f: feat_dict_base.get(f, 0.0) for f in model_feats}])
        if best_name == "XGBoost":
            base_log = float(best_m.predict(row_base)[0])
        else:
            base_log = float(best_m.predict(sim_scaler.transform(row_base))[0])
        base_rate = float(np.expm1(base_log))

        reg_pop = float(population[(population["Region"] == sim_region) & (population["Year"] == YEAR_MAX)]["Total_population"].values[0]) if not population[(population["Region"] == sim_region) & (population["Year"] == YEAR_MAX)].empty else 50000.0
        pred_incidents = int(round((pred_rate / 100000.0) * reg_pop))
        prevented_incidents = int(round(((base_rate - pred_rate) / 100000.0) * reg_pop))

        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">Projected Monthly Assault Rate</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{pred_rate:.1f} <span style="font-size:1.1rem;color:#718096;">per 100k</span></div>', unsafe_allow_html=True)
        st.markdown(f"**Expected Monthly Incidents**: ~{pred_incidents:,} assaults in {sim_region}")
        if pac_curb != 0:
            st.markdown(f"**Difference from the no-change scenario**: about **{-prevented_incidents:+,}** assaults per month "
                        f"under the assumed factors. This is a sensitivity result, not an estimate of what a policy would achieve.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(
            "Illustrative scenario only. The 0.4 and 0.2 pass-through factors are assumptions, "
            "not estimates, and the model captures association rather than cause."
        )
        with st.expander("Assumptions behind this scenario"):
            st.markdown(f"""
            - A change of **x%** in wholesale alcohol supply is **assumed** to change last month's alcohol-involved assaults by **0.4x%**
              and domestic-violence assaults by **0.2x%**. These factors are chosen for illustration; the project did not estimate them.
            - The model is the saved **{best_name}** from Model A (`Models/model_a.joblib`, trained {sim_period}).
              The lag-1, lag-3 and lag-12 rates are all set to the value entered above.
            - Alcohol supply was positively associated with assault rates but did **not** improve forecast accuracy (report, item 5.3),
              so the output shows how sensitive the model is to the assumption, not the effect of a policy.
            - No staffing or policy action should be taken from this scenario without validation with PFES.
            """)
        st.write("")

        # Tier of the scenario rate (descriptive; no action is prescribed)
        if pred_rate >= rate_threshold:
            st.markdown('<div class="alert-critical">🔴 <b>Scenario rate above the planning threshold</b></div>', unsafe_allow_html=True)
        elif pred_rate >= rate_threshold * 0.7:
            st.markdown('<div class="alert-elevated">⚠️ <b>Scenario rate within 30% of the threshold</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-baseline">✅ <b>Scenario rate below the planning threshold</b></div>', unsafe_allow_html=True)


# =============================================================================
# TAB 7: Briefing Note Export
# =============================================================================
with tabs[6]:
    st.subheader("📥 Export Executive Briefing & Data Extracts")
    st.markdown("Generate formatted tables and extracts for ministerial briefings, cabinet submissions, and police briefings.")

    e_col1, e_col2 = st.columns([3, 1])
    with e_col1:
        st.markdown("##### Filtered Extract Preview")
        cols_to_export = ["Year", "Quarter", "Month number", "Region", "Offence category",
                          "Number of offences", "Alcohol_offences", "DV_offences", "Total_population"]
        st.dataframe(filtered_panel[cols_to_export].head(250), use_container_width=True)

    with e_col2:
        st.write("")
        st.write("")
        csv_data = filtered_panel[cols_to_export].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download Cabinet CSV Extract",
            data=csv_data,
            file_name=f"nt_crime_briefing_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv"
        )

