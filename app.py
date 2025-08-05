# === 1. Imports and Setup ===
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Player Readiness",
    page_icon="BostonBoltsLogo.png",
    layout="wide"
)

# === HELPERS ===
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Start Date'], format='%m/%d/%y', errors='coerce')
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
    return df

def get_color(ratio):
    if ratio < 0.5: return "red"
    if ratio < 0.75: return "orange"
    if ratio < 1.0: return "yellow"
    if ratio <= 1.30: return "green"
    return "black"

def create_readiness_gauge(value, benchmark, label):
    ratio = 0 if pd.isna(benchmark) or benchmark == 0 else value / benchmark
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(ratio, 2),
        number={"font": {"size": 20}},
        gauge={
            "axis": {"range": [0, max(1.5, ratio)], "showticklabels": False},
            "bar": {"color": get_color(ratio)},
            "steps": [
                {"range": [0, 0.5], "color": "#ffcccc"},
                {"range": [0.5, 0.75], "color": "#ffe0b3"},
                {"range": [0.75, 1.0], "color": "#ffffcc"},
                {"range": [1.0, 1.3], "color": "#ccffcc"},
                {"range": [1.3, max(1.5, ratio)], "color": "#e6e6e6"}
            ]
        }
    ))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=180)
    return fig

# === METRICS ===
METRICS = [
    "Distance (m)",
    "High Intensity Running (m)",
    "Sprint Distance (m)",
    "No. of Sprints",
    "Top Speed (kph)"
]
METRIC_LABELS = {
    "Distance (m)": "Total Distance",
    "High Intensity Running (m)": "HSR",
    "Sprint Distance (m)": "Sprint Distance",
    "No. of Sprints": "# of Sprints",
    "Top Speed (kph)": "Top Speed"
}

# === SESSION-STATE INIT ===
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "proceed" not in st.session_state:
    st.session_state.proceed = False
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# === LOGO & TITLE ===
with st.container():
    c1, c2, c3 = st.columns([0.08, 0.001, 0.72])
    with c1: st.image("BostonBoltsLogo.png", width=120)
    with c2: st.markdown("<div style='border-left:2px solid gray; height:90px;'></div>", unsafe_allow_html=True)
    with c3: st.image("MLSNextLogo.png", width=120)
with st.container():
    st.markdown(
        "<h1 style='text-align:center;font-size:72px;margin-top:-60px;'>Player Readiness</h1>",
        unsafe_allow_html=True
    )

# === LANDING PAGE ===
if st.session_state.page == "Home":
    st.markdown("### Select a Dashboard")
    choice = st.selectbox(
        "Choose which dashboard you want to view:",
        ["Player Gauges Dashboard", "ACWR Dashboard"]
    )
    if st.button("Continue", key="dashboard_continue"):
        st.session_state.page = choice
        st.session_state.proceed = False
        st.rerun()
    st.stop()

# === PLAYER GAUGES DASHBOARD ===
if st.session_state.page == "Player Gauges Dashboard":
    # Debug toggle
    dbg_label = "Hide Debug Info" if st.session_state.show_debug else "Show Debug Info"
    if st.button(dbg_label, key="debug_toggle"):
        st.session_state.show_debug = not st.session_state.show_debug
        st.rerun()

    st.markdown("## Player Gauges Dashboard")

    # Team selection
    if not st.session_state.proceed:
        teams = [f"U{age} MLS Next" for age in [15,16,17,19]] + [f"U{age} MLS Next 2" for age in [15,16,17,19]]
        sel = st.selectbox("Select Team", teams, key="gauges_team")
        if st.button("Continue", key="gauges_continue"):
            st.session_state.proceed = True
            st.session_state.selected_team = sel
            st.rerun()
        st.stop()

    # Back to landing
    if st.button("Select Dashboard", key="gauges_back"):
        st.session_state.page = "Home"
        st.session_state.proceed = False
        st.rerun()

    # Load and prep data
    team = st.session_state.selected_team
    path = f"Player Data/{team}_PD_Data.csv"
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()
    df = load_data(path)
    df = df.dropna(subset=["Date","Session Type","Athlete Name","Segment Name"])
    df = df[df["Segment Name"]=="Whole Session"].sort_values("Date")

    # Anchors
    match_df = df[df["Session Type"]=="Match Session"]
    if match_df.empty:
        st.markdown("**Latest Match Date Used:** _None found_")
        st.stop()
    latest_match_date = match_df["Date"].max()
    st.markdown(f"**Latest Match Date Used:** `{latest_match_date.date()}`")

    train_df = df[df["Session Type"]=="Training Session"]
    if train_df.empty:
        st.warning("No training sessions found.")
        st.stop()
    latest_training_date = train_df["Date"].max()
    iso_year, iso_week, _ = latest_training_date.isocalendar()
    st.markdown(f"🌐 Global Latest Training Date: {latest_training_date.date()}")

    # Loop players & render gauges
    for player in sorted(df["Athlete Name"].dropna().unique()):
        p_df = df[df["Athlete Name"]==player].copy()
        p_df["Duration (mins)"] = pd.to_numeric(p_df["Duration (mins)"], errors="coerce")
        for m in METRICS:
            p_df[m] = pd.to_numeric(p_df[m], errors="coerce")

        matches = p_df[
            (p_df["Session Type"]=="Match Session") &
            (p_df["Date"]<=latest_match_date) &
            (p_df["Duration (mins)"]>0)
        ].sort_values("Date")
        if matches.empty:
            continue

        iso = p_df["Date"].dt.isocalendar()
        training_week = p_df[
            (p_df["Session Type"]=="Training Session") &
            (iso["week"]==iso_week) &
            (iso["year"]==iso_year)
        ]
        if training_week.empty:
            training_week = pd.DataFrame([{
                "Athlete Name": player,
                "Date": latest_training_date,
                "Session Type": "Training Session",
                "Segment Name": "Whole Session",
                **{m: 0 for m in METRICS},
                "Duration (mins)": 0
            }])

        prev = p_df[
            (p_df["Session Type"]=="Training Session") &
            (p_df["Date"]<training_week["Date"].min())
        ]
        if not prev.empty:
            prev["Year"] = prev["Date"].dt.isocalendar().year
            prev["Week"] = prev["Date"].dt.isocalendar().week
            week_sums = prev.groupby(["Year","Week"])[METRICS].sum().reset_index()
            valid = week_sums[(week_sums.drop(columns=["Year","Week"])>0).any(axis=1)]
            if not valid.empty:
                last = valid.iloc[-1]
                prev_week_str = f"Week {int(last['Week'])}, {int(last['Year'])}"
                prev_data = prev[
                    (prev["Date"].dt.isocalendar().week==last["Week"]) &
                    (prev["Date"].dt.isocalendar().year==last["Year"])
                ]
                previous_week_total_map = {
                    m: prev_data[m].sum() for m in METRICS if m!="Top Speed (kph)"
                }
            else:
                prev_week_str="None"
                previous_week_total_map={m:0 for m in METRICS if m!="Top Speed (kph)"}
        else:
            prev_week_str="None"
            previous_week_total_map={m:0 for m in METRICS if m!="Top Speed (kph)"}

        match_avg = {}
        for m in METRICS:
            if m!="Top Speed (kph)":
                matches["Per90"] = matches[m]/matches["Duration (mins)"]*90
                match_avg[m] = matches["Per90"].mean()

        top_speed_benchmark = p_df["Top Speed (kph)"].max()
        grouped = training_week.agg({
            "Distance (m)":"sum",
            "High Intensity Running (m)":"sum",
            "Sprint Distance (m)":"sum",
            "No. of Sprints":"sum",
            "Top Speed (kph)":"max"
        }).to_frame().T

        st.markdown(f"### {player}")
        cols = st.columns(len(METRICS))

        for i, metric in enumerate(METRICS):
            if metric=="Top Speed (kph)":
                train_val = grouped[metric].max()
                benchmark = top_speed_benchmark
                ratio = 0 if pd.isna(benchmark) or benchmark==0 else train_val/benchmark

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(ratio,2),
                    number={"font":{"size":20}},
                    gauge={
                        "axis":{"range":[0,1.0],"showticklabels":False},
                        "bar":{"color":
                            "red"    if ratio<0.5 else
                            "orange" if ratio<0.75 else
                            "yellow" if ratio<0.9 else
                            "green"
                        },
                        "steps":[
                            {"range":[0,0.5],"color":"#ffcccc"},
                            {"range":[0.5,0.75],"color":"#ffe0b3"},
                            {"range":[0.75,0.9],"color":"#ffffcc"},
                            {"range":[0.9,1.0],"color":"#ccffcc"},
                        ]
                    }
                ))
                fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=180)
                with cols[i]:
                    st.markdown(f"<div style='text-align:center;font-weight:bold;'>{METRIC_LABELS[metric]}</div>", unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, key=f"{player}-top-{i}")
                    if ratio < 0.9:
                        st.markdown(
                            "<div style='text-align:center;color:red;font-weight:bold;'>"
                            "⚠️ Did not reach 90% of max speed this week"
                            "</div>",
                            unsafe_allow_html=True
                        )
                continue

            train_val = grouped[metric].sum()
            benchmark = match_avg.get(metric, None)
            fig = create_readiness_gauge(train_val, benchmark, METRIC_LABELS[metric])
            with cols[i]:
                st.markdown(f"<div style='text-align:center;font-weight:bold;'>{METRIC_LABELS[metric]}</div>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, key=f"{player}-{metric}-{i}")

                practices_done = training_week.shape[0]
                current_sum = training_week[metric].sum()
                previous_week_total = previous_week_total_map.get(metric, 0)

                iso_all = p_df["Date"].dt.isocalendar()
                p_df["PracticeNumber"] = (
                    p_df[p_df["Session Type"]=="Training Session"]
                    .groupby([iso_all.year, iso_all.week]).cumcount()+1
                ).clip(upper=3)
                practice_avgs = (
                    p_df[p_df["Session Type"]=="Training Session"]
                    .groupby("PracticeNumber")[metric].mean()
                    .reindex([1,2,3], fill_value=0)
                )

                if previous_week_total>0 and current_sum>1.10*previous_week_total:
                    flag="⚠️"; flag_val=current_sum; projection_used=False; projected_total=None
                else:
                    if practices_done<3:
                        needed = list(range(practices_done+1,4))
                        projected_total = current_sum + practice_avgs.loc[needed].sum()
                        flag_val=projected_total; projection_used=True
                    else:
                        projected_total=None; flag_val=current_sum;_projection_used=False
                    if previous_week_total>0 and flag_val>1.10*previous_week_total:
                        flag="🔮⚠️" if projection_used else "⚠️"
                    else:
                        flag=""

                if flag:
                    if projection_used:
                        st.markdown(
                            "<div style='text-align:center;font-weight:bold;'>"
                            f"⚠️Projected total of {METRIC_LABELS[metric]} is on track to be > 110% of last week’s total"
                            "</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            "<div style='text-align:center;font-weight:bold;'>"
                            f"⚠️{METRIC_LABELS[metric]} is > 110% than last week’s total"
                            "</div>",
                            unsafe_allow_html=True
                        )

                if st.session_state.show_debug:
                    st.markdown(f"""
                        <div style='font-size:14px;color:#555;'>
                            <b>Debug for {METRIC_LABELS[metric]}</b><br>
                            • Previous Week Used: {prev_week_str}<br>
                            • Previous Week Total: {previous_week_total:.1f}<br>
                            • Current Week So Far: {current_sum:.1f}<br>
                            • Practices Done: {practices_done}<br>
                            • Historical Practice Avgs: {practice_avgs.to_dict()}<br>
                            • Projected Total: {projected_total if projection_used else 'N/A'}<br>
                            • Final Used: {flag_val:.1f} ({'Projected' if projection_used else 'Actual'})<br>
                            • Threshold (110%): {1.10*previous_week_total:.1f}<br>
                            • ⚠️ Flag: {'YES' if flag else 'NO'}
                        </div>
                    """, unsafe_allow_html=True)

# === ACWR DASHBOARD ===
if st.session_state.page == "ACWR Dashboard":
    st.markdown("## ACWR Dashboard")

    # --- STEP 1: pick team ---
    if not st.session_state.proceed:
        teams = [f"U{age} MLS Next" for age in [15,16,17,19]] + \
                [f"U{age} MLS Next 2" for age in [15,16,17,19]]
        sel = st.selectbox("Select Team", teams, key="acwr_team")
        if st.button("Continue", key="acwr_continue"):
            st.session_state.proceed = True
            st.session_state.selected_team = sel
            st.rerun()
        st.stop()

    # --- BACK TO HOME ---
    if st.button("Select Dashboard", key="acwr_back"):
        st.session_state.page = "Home"
        st.session_state.proceed = False
        st.rerun()

    # --- LOAD & FILTER DATA ---
    team = st.session_state.selected_team
    df_acwr = load_data(f"Player Data/{team}_PD_Data.csv")
    df_acwr = (
        df_acwr
        .dropna(subset=["Date","Session Type","Athlete Name","Segment Name"])
        .query("`Segment Name`=='Whole Session' and `Session Type`=='Training Session'")
    )
    df_acwr['Date'] = pd.to_datetime(df_acwr['Date'])

    # --- DAILY AGGREGATION ---
    metrics_acwr = ["Distance (m)", "High Intensity Running (m)",
                    "Sprint Distance (m)", "No. of Sprints"]
    df_daily = (
        df_acwr
        .groupby(["Athlete Name","Date"])[metrics_acwr]
        .sum()
        .reset_index()
        .sort_values(["Athlete Name","Date"])
        .set_index("Date")
    )

    # --- ROLLING ACUTE/CHRONIC & ACWR ---
    for m in metrics_acwr:
        df_daily[f"acute_{m}"] = (
            df_daily.groupby("Athlete Name")[m]
                    .rolling("7d").sum()
                    .reset_index(0, drop=True)
        )
        df_daily[f"chronic_{m}"] = (
            df_daily.groupby("Athlete Name")[m]
                    .rolling("28d").sum()
                    .reset_index(0, drop=True) / 4
        )
        df_daily[f"acwr_{m}"] = df_daily[f"acute_{m}"] / df_daily[f"chronic_{m}"]

    df_daily = df_daily.reset_index()

    # --- COLOR MAP & TOGGLES ---
    color_map = {
        "Distance (m)": "#1f77b4",
        "High Intensity Running (m)": "#ff7f0e",
        "Sprint Distance (m)": "#2ca02c",
        "No. of Sprints": "#d62728"
    }
    st.markdown("#### Show / Hide Metrics")
    cols = st.columns(len(metrics_acwr))
    show_metric = {}
    for idx, m in enumerate(metrics_acwr):
        show_metric[m] = cols[idx].checkbox(METRIC_LABELS[m], True, key=f"show_{m}")

    active_metrics = [m for m in metrics_acwr if show_metric[m]]
    if not active_metrics:
        st.info("Select at least one metric to display.")
    else:
        st.markdown("### Daily Rolling ACWR (7d ∶ 28d) by Player")
        for player in sorted(df_daily["Athlete Name"].unique()):
            p = df_daily[df_daily["Athlete Name"] == player]
            if p.empty: continue

            fig = go.Figure()
            for m in active_metrics:
                fig.add_trace(go.Scatter(
                    x=p["Date"],
                    y=p[f"acwr_{m}"],
                    mode="lines+markers",
                    name=METRIC_LABELS[m],
                    line_shape="spline",
                    line=dict(width=2, color=color_map[m]),
                    marker=dict(size=4, color=color_map[m])
                ))

            # highlight sweet-spot band
            fig.add_hrect(y0=0.8, y1=1.3, fillcolor="lightgreen", opacity=0.2, line_width=0)

            fig.update_layout(
                template="plotly_white",
                title=f"{player} — ACWR (7d ∶ 28d)",
                xaxis_title="Date",
                yaxis_title="ACWR",
                legend_title="Metric",
                font=dict(family="Arial", size=12),
                margin=dict(t=50, b=40, l=40, r=40),
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- WARNINGS ---
            warnings = []
            for m in active_metrics:
                last_series = p[f"acwr_{m}"].dropna()
                if not last_series.empty:
                    val = last_series.iloc[-1]
                    if val < 0.8:
                        warnings.append(f"🚨 Undertrained for {METRIC_LABELS[m]}: ACWR = {val:.2f}")
                    elif val > 1.3:
                        warnings.append(f"⚠️ Overtrained for {METRIC_LABELS[m]}: ACWR = {val:.2f}")
            for w in warnings:
                st.markdown(w)