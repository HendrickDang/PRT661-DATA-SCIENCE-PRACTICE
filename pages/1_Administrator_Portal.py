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
    ASSAULT_CATEGORY,
    KEY_TOWNS,
    MONTH_LABELS,
    REGION_COLORS,
    REGION_GEO,
    REGION_ORDER,
    YEAR_MAX,
    YEAR_MIN,
    get_regression_panel,
    load_data,
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
    }
    .alert-elevated {
        background-color: #fffaf0;
        border-left: 5px solid #dd6b20;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    .alert-baseline {
        background-color: #f0fff4;
        border-left: 5px solid #38a169;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="admin-header">🏛️ Northern Territory Justice & Public Safety Resource Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="admin-sub">Executive Decision Support for Frontline Policing, Health Services, and Policy Planning</div>', unsafe_allow_html=True)

# Load data
panel, population, monthly_assault = load_data()
reg_panel, train, test, dummy_cols, cv_frame = get_regression_panel(monthly_assault, population)
model_artifacts = run_model_training(train, test, cv_frame, dummy_cols)

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
    "Critical Assault Rate Threshold (/100k / month)",
    min_value=150, max_value=700, value=350, step=25,
    help="Per-capita rate that activates frontline surge deployment protocols."
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
    "🗺️ Crime Map & Spatial Resource Demand",
    "🚨 Resource Priority & Staffing Matrix",
    "📅 Seasonal Surge Calendar",
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
    latest_pop = population[population["Year"] == year_range[1]]["Total_population"].sum()

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
        st.markdown("#### 📌 Key Policy Takeaways")
        st.markdown(f"""
        - **Assault is the #1 Crime Challenge**: Violent crime accounts for the largest demand on Northern Territory Police and hospital emergency departments.
        - **High Alcohol & Domestic Violence Nexus**: **{alc_assault_share:.1f}%** of all recorded assaults are alcohol-involved, and **{dv_assault_share:.1f}%** involve domestic or family violence. Policy interventions curbing secondary liquor supply yield direct reductions in violent incidents.
        - **Per-Capita Disparity**: Greater Darwin generates the largest absolute headcount of incidents due to its population size (over 60% of NT residents), but remote regional jurisdictions (such as Central Australia and Barkly) suffer from significantly higher per-capita assault rates.
        """)

    with c_right:
        st.markdown("#### ⚡ Frontline Alert Summary")
        latest_year = year_range[1]
        latest_rates = filtered_assault[filtered_assault["Year"] == latest_year].groupby("Region")["Assault_rate_100k"].mean()
        critical_regs = latest_rates[latest_rates >= rate_threshold].index.tolist()
        elevated_regs = latest_rates[(latest_rates < rate_threshold) & (latest_rates >= rate_threshold * 0.7)].index.tolist()

        if critical_regs:
            st.markdown(f'<div class="alert-critical">🚨 <b>CRITICAL SURGE REQUIRED</b><br>'
                        f'Regions exceeding alert threshold ({rate_threshold}/100k): <b>{", ".join(critical_regs)}</b>.<br>'
                        f'Immediate priority for mobile police patrols and emergency domestic violence shelter capacity.</div>', unsafe_allow_html=True)
        if elevated_regs:
            st.markdown(f'<div class="alert-elevated">⚠️ <b>ELEVATED MONITORING</b><br>'
                        f'Regions in elevated tier: <b>{", ".join(elevated_regs)}</b>.<br>'
                        f'Targeted liquor licensing enforcement recommended.</div>', unsafe_allow_html=True)
        if not critical_regs and not elevated_regs:
            st.markdown('<div class="alert-baseline">✅ <b>BASELINE OPERATIONAL TIER</b><br>All selected jurisdictions are operating within standard historical limits.</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2: Spatial Crime Map & Resource Demand
# =============================================================================
with tabs[1]:
    st.subheader("🗺️ Northern Territory Spatial Crime Distribution")
    st.markdown(
        "Visualise crime density across the Territory. "
        "Toggle between **Per-Capita Victimisation Rate** (which region needs services most urgently) "
        "and **Total Incident Headcount** (where the raw volume of police officers is deployed)."
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
        "**Crucial for Cabinet & Budget Allocation**: Allocating staff purely by raw headcount over-allocates to Greater Darwin "
        "while under-funding Central Australia and Barkly where per-capita violence is much higher."
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
# TAB 3: Resource Priority & Staffing Matrix
# =============================================================================
with tabs[2]:
    st.subheader("🚨 Frontline Staffing & Operational Allocation Matrix")
    st.markdown(
        "This decision matrix maps regional risk profiles to clear operational recommendations "
        "for police roster managers, ambulance deployment, and crisis accommodation providers."
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
            return "URGENT SURGE: Deploy mobile frontline police, expand night crisis shelter, surge hospital ED night staff."
        elif rate >= rate_threshold * 0.7:
            return "ELEVATED ROSTER: Increase targeted liquor compliance checks, maintain active domestic violence outreach."
        else:
            return "STANDARD POSTING: Maintain baseline community policing and primary healthcare schedules."

    reg_summary["Alert Status"] = reg_summary["Avg_Monthly_Rate"].apply(
        lambda r: "🔴 CRITICAL" if r >= rate_threshold else ("🟠 ELEVATED" if r >= rate_threshold * 0.7 else "🟢 BASELINE")
    )
    reg_summary["Recommended Operational Response"] = reg_summary.apply(determine_action, axis=1)
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
            "Recommended Operational Response": "Prescribed Resource Action"
        },
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# TAB 4: Seasonal Surge Calendar
# =============================================================================
with tabs[3]:
    st.subheader("📅 Northern Territory Seasonal Surge Calendar")
    st.markdown(
        "Crime in the Northern Territory follows a pronounced seasonal rhythm. "
        "The **Wet Season (November to April)** historically brings elevated assault volume due to temperature, "
        "holiday periods, and community mobility patterns."
    )

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        # Monthly Seasonality
        monthly_season = filtered_assault.groupby("Month number", as_index=False)["Assault_offences"].sum()
        monthly_season["Month"] = monthly_season["Month number"].map(lambda m: MONTH_LABELS[m - 1])
        monthly_season["Season"] = monthly_season["Month number"].apply(lambda m: "Wet Season (Surge)" if m in [11, 12, 1, 2, 3, 4] else "Dry Season")

        fig_season = px.bar(
            monthly_season,
            x="Month",
            y="Assault_offences",
            color="Season",
            color_discrete_map={"Wet Season (Surge)": "#EF5350", "Dry Season": "#42A5F5"},
            title="Total Assault Volume by Month (Wet vs Dry Season)",
            labels={"Assault_offences": "Total Assaults"}
        )
        fig_season.update_layout(height=380)
        st.plotly_chart(fig_season, use_container_width=True)

    with s_col2:
        st.markdown("##### 📋 Recommended Seasonal Staffing Schedule:")
        st.markdown("""
        - **October Pre-Surge**: Complete leave rotations and ensure emergency department staffing is finalized before the November monsoon onset.
        - **November – January (Peak Surge)**: Implement maximum mobile police rosters, enforce Banned Drinker Register (BDR) compliance, and open supplementary crisis shelter beds.
        - **February – April**: Monitor secondary surge following holiday periods; focus on youth outreach and community transport.
        - **May – September (Dry Season Reset)**: Baseline demand period; optimal window for officer training, fleet maintenance, and regional infrastructure upgrades.
        """)

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
# TAB 5: Policy Intervention Simulator
# =============================================================================
with tabs[4]:
    st.subheader("🎯 Policy Intervention & Scenario Simulator")
    st.markdown(
        "Test real-world policy levers: **Simulate the expected impact of wholesale alcohol supply restrictions "
        "(e.g., takeaway curbs, BDR enforcement) on projected assault rates and frontline alerts.**"
    )

    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        st.markdown("##### 1. Jurisdiction & Environmental Context")
        sim_region = st.selectbox("Target Jurisdiction:", options=REGION_ORDER, index=0)
        sim_month = st.slider("Target Forecast Month:", min_value=1, max_value=12, value=12,
                              format="%d - " + MONTH_LABELS[11])
        is_wet = 1 if sim_month in [11, 12, 1, 2, 3, 4] else 0
        st.caption(f"Climate Period: **{'Wet Season (Monsoonal Peak)' if is_wet else 'Dry Season'}**")

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
        st.markdown("##### 📊 Projected Operational Outcome")

        # Feature vector
        sin_m = np.sin(2 * np.pi * sim_month / 12)
        model_feats = model_artifacts["features"]
        best_name = model_artifacts["best_name"]
        best_m = model_artifacts["results"][best_name]["model"]

        # Predict with simulated policy curb
        feat_dict = {
            "sin_month": sin_m,
            "Season": is_wet,
            "assault_rate_lag1": sim_lag1,
            "assault_rate_lag12": sim_lag1,
            "Alcohol_offences": def_alc_off * (1.0 + (pac_curb / 100.0) * 0.4),
            "DV_offences": def_dv_off * (1.0 + (pac_curb / 100.0) * 0.2),
        }
        for r in REGION_ORDER:
            if r != "Greater Darwin":
                feat_dict[f"Reg_{r}"] = 1 if sim_region == r else 0

        row_input = pd.DataFrame([{f: feat_dict.get(f, 0.0) for f in model_feats}])
        if best_name == "XGBoost":
            pred_log = float(best_m.predict(row_input)[0])
        else:
            pred_log = float(best_m.predict(model_artifacts["scaler"].transform(row_input))[0])
        pred_rate = float(np.expm1(pred_log))

        # Predict baseline without policy curb
        feat_dict_base = {
            "sin_month": sin_m,
            "Season": is_wet,
            "assault_rate_lag1": sim_lag1,
            "assault_rate_lag12": sim_lag1,
            "Alcohol_offences": def_alc_off,
            "DV_offences": def_dv_off,
        }
        for r in REGION_ORDER:
            if r != "Greater Darwin":
                feat_dict_base[f"Reg_{r}"] = 1 if sim_region == r else 0

        row_base = pd.DataFrame([{f: feat_dict_base.get(f, 0.0) for f in model_feats}])
        if best_name == "XGBoost":
            base_log = float(best_m.predict(row_base)[0])
        else:
            base_log = float(best_m.predict(model_artifacts["scaler"].transform(row_base))[0])
        base_rate = float(np.expm1(base_log))

        reg_pop = float(population[(population["Region"] == sim_region) & (population["Year"] == YEAR_MAX)]["Total_population"].values[0]) if not population[(population["Region"] == sim_region) & (population["Year"] == YEAR_MAX)].empty else 50000.0
        pred_incidents = int(round((pred_rate / 100000.0) * reg_pop))
        prevented_incidents = int(round(((base_rate - pred_rate) / 100000.0) * reg_pop))

        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">Projected Monthly Assault Rate</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{pred_rate:.1f} <span style="font-size:1.1rem;color:#718096;">per 100k</span></div>', unsafe_allow_html=True)
        st.markdown(f"**Expected Monthly Incidents**: ~{pred_incidents:,} assaults in {sim_region}")
        if pac_curb < 0:
            st.markdown(f"🎉 **Estimated Harm Reduction**: **~{prevented_incidents:,} fewer assault victims** per month from the policy intervention.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")

        # Actionable Recommendation
        if pred_rate >= rate_threshold:
            st.markdown('<div class="alert-critical">🚨 <b>CRITICAL SURGE TIER</b><br>'
                        'Projected rate exceeds alert threshold. Mandate maximum frontline patrol deployment and standby emergency shelter intake.</div>', unsafe_allow_html=True)
        elif pred_rate >= rate_threshold * 0.7:
            st.markdown('<div class="alert-elevated">⚠️ <b>ELEVATED RISK TIER</b><br>'
                        'Moderate surge anticipated. Target point-of-sale liquor enforcement.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-baseline">✅ <b>BASELINE OPERATIONAL TIER</b><br>'
                        'Standard staffing allocation is sufficient.</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 6: Briefing Note Export
# =============================================================================
with tabs[5]:
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

