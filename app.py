import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google import genai
from google.genai import types

BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "analysis" / "outputs"
FIGS = BASE / "analysis" / "figures_final"

st.set_page_config(page_title="UIDAI Aadhaar Insights", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.0rem; padding-bottom: 3rem;}
    .stTabs [data-baseweb="tab-list"] button {font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUTPUTS / f"{name}.parquet")


@st.cache_data
def load_summary() -> dict:
    with open(OUTPUTS / "summary_metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_geojson(city: str) -> dict | None:
    geo_path = OUTPUTS / f"{city}_pincode.geojson"
    if not geo_path.exists():
        return None
    with open(geo_path) as f:
        return json.load(f)


@st.cache_data
def load_state_geojson() -> dict | None:
    geo_path = OUTPUTS / "india_states.geojson"
    if not geo_path.exists():
        return None
    with open(geo_path) as f:
        return json.load(f)


summary = load_summary()
state_daily = load_parquet("state_daily")
state_monthly = load_parquet("state_monthly")
state_summary = load_parquet("state_summary")
district_focus = load_parquet("district_focus")
district_summary = load_parquet("district_summary")
dow_summary = load_parquet("dow_summary")
monthly_summary = load_parquet("monthly_summary")
pincode_anomalies = load_parquet("pincode_anomalies")

state_list = sorted(state_summary["state"].unique().tolist())

# Sidebar controls
st.sidebar.header("Filters")
selected_state = st.sidebar.selectbox("State", ["All States"] + state_list, index=0)
selected_city = st.sidebar.selectbox(
    "City Map",
    ["Bangalore", "Mumbai", "Chennai", "Hyderabad", "Kolkata", "Ahmedabad"],
    index=0,
)
metric_choice = st.sidebar.radio("City Map Metric", ["Updates", "Enrolments"], horizontal=True)

st.title("UIDAI Aadhaar Enrolment & Update Intelligence")
st.caption(
    "Coverage: {start} to {end}. Daily patterns are computed on dense months: {dense}"
    .format(start=summary["date_min"], end=summary["date_max"], dense=", ".join(summary["daily_focus_months"]))
)

# KPI row
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Total Enrolments", f"{summary['total_enrolments']:,}")
with kpi_cols[1]:
    st.metric("Total Updates", f"{summary['total_demographic_updates'] + summary['total_biometric_updates']:,}")
with kpi_cols[2]:
    st.metric("Biometric Share", f"{summary['total_biometric_updates'] / (summary['total_demographic_updates'] + summary['total_biometric_updates']):.1%}")
with kpi_cols[3]:
    st.metric("Updates / Enrolment", f"{summary['overall_update_ratio']:.1f}x")


tabs = st.tabs([
    "Overview",
    "India Map",
    "Trends",
    "State Deep Dive",
    "District Insights",
    "Pincode Anomalies",
    "City Maps",
    "AI Analyst",
    "Downloads",
])

# Overview
with tabs[0]:
    st.subheader("National Overview")

    # Monthly trend chart
    monthly = monthly_summary.sort_values("month")
    fig = go.Figure()
    fig.add_bar(x=monthly["month"], y=monthly["demo_total"], name="Demographic Updates", marker_color="#ff7f0e")
    fig.add_bar(x=monthly["month"], y=monthly["bio_total"], name="Biometric Updates", marker_color="#2ca02c")
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["enrol_total"], name="Enrolment", marker_color="#1f77b4",
        mode="lines+markers",
    ))
    fig.update_layout(barmode="stack", height=420, legend_orientation="h")
    fig.update_yaxes(title_text="Transactions")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Update Mix by State")
        state_summary["demo_to_bio_ratio"] = state_summary["demo_total"] / state_summary["bio_total"].replace(0, np.nan)
        fig_mix = px.scatter(
            state_summary,
            x="demo_to_bio_ratio",
            y="update_total",
            text="state",
            color="update_total",
            color_continuous_scale="Viridis",
        )
        fig_mix.update_traces(textposition="top center")
        fig_mix.update_layout(height=420, coloraxis_showscale=False)
        fig_mix.update_xaxes(title="Demographic / Biometric Ratio")
        fig_mix.update_yaxes(title="Total Updates")
        st.plotly_chart(fig_mix, use_container_width=True)

    with col2:
        st.subheader("Age Mix (Top 12 States)")
        age = state_summary.sort_values("enrol_total", ascending=False).head(12).copy()
        age["age_0_5_share"] = age["age_0_5"] / age["enrol_total"].replace(0, np.nan)
        age["age_5_17_share"] = age["age_5_17"] / age["enrol_total"].replace(0, np.nan)
        age["age_18_share"] = age["age_18_greater"] / age["enrol_total"].replace(0, np.nan)
        fig_age = go.Figure()
        fig_age.add_bar(x=age["state"], y=age["age_0_5_share"], name="0-5")
        fig_age.add_bar(x=age["state"], y=age["age_5_17_share"], name="5-17")
        fig_age.add_bar(x=age["state"], y=age["age_18_share"], name="18+")
        fig_age.update_layout(barmode="stack", height=420, legend_orientation="h")
        fig_age.update_yaxes(title="Share of Enrolments", tickformat=".0%")
        st.plotly_chart(fig_age, use_container_width=True)

# India Map
with tabs[1]:
    st.subheader("India State Map")
    state_geo = load_state_geojson()
    if state_geo is None:
        st.warning("State geojson not found. Run analysis/run_analysis.py to generate it.")
    else:
        state_df = pd.DataFrame([feat["properties"] for feat in state_geo["features"]])
        for col in ["enrol_total", "update_total", "update_enrol_ratio", "updates_per_1k", "bio_share"]:
            if col in state_df.columns:
                state_df[col] = pd.to_numeric(state_df[col], errors="coerce")

        metric_labels = {
            "Total Updates": "update_total",
            "Total Enrolments": "enrol_total",
            "Updates per 1k": "updates_per_1k",
            "Update/Enrol Ratio": "update_enrol_ratio",
            "Biometric Share": "bio_share",
        }
        metric_label = st.selectbox("Metric", list(metric_labels.keys()), index=0)
        metric_col = metric_labels[metric_label]

        fig_map = px.choropleth(
            state_df,
            geojson=state_geo,
            locations="state",
            featureidkey="properties.state",
            color=metric_col,
            color_continuous_scale="YlOrRd",
            hover_data=["state", "update_total", "enrol_total", "update_enrol_ratio", "updates_per_1k", "bio_share"],
            labels={metric_col: metric_label},
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(height=620, margin={"r": 0, "t": 10, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)

        rank_df = state_df.sort_values(metric_col, ascending=False).head(15)
        fig_rank = px.bar(
            rank_df,
            x=metric_col,
            y="state",
            orientation="h",
            labels={metric_col: metric_label, "state": "State"},
            color=metric_col,
            color_continuous_scale="Viridis",
        )
        fig_rank.update_layout(height=520, coloraxis_showscale=False)
        st.plotly_chart(fig_rank, use_container_width=True)

        def fmt_value(val: float) -> str:
            if metric_col == "bio_share":
                return f"{val:.1%}"
            if metric_col in {"update_enrol_ratio", "updates_per_1k"}:
                return f"{val:.1f}"
            return f"{val:,.0f}"

        top_row = rank_df.iloc[0]
        bottom_row = state_df.sort_values(metric_col, ascending=True).iloc[0]
        share_top5 = None
        if metric_col in {"update_total", "enrol_total"} and state_df[metric_col].sum() > 0:
            share_top5 = rank_df.head(5)[metric_col].sum() / state_df[metric_col].sum()

        k1, k2, k3 = st.columns(3)
        k1.metric("Top State", f"{top_row['state']}", fmt_value(top_row[metric_col]))
        k2.metric("Bottom State", f"{bottom_row['state']}", fmt_value(bottom_row[metric_col]))
        if share_top5 is not None:
            k3.metric("Top 5 Share", f"{share_top5:.1%}")
        else:
            k3.metric("Top 5 Avg", fmt_value(rank_df.head(5)[metric_col].mean()))

        with st.expander("Gemini map insights"):
            map_model = st.selectbox(
                "Model",
                ["gemini-2.5-flash-lite", "gemini-1.5-flash", "gemini-2.0-flash-exp"],
                index=0,
                key="map_model",
            )
            custom_q = st.text_input("Ask about this map (optional)", key="map_question")
            if st.button("Generate Insight", key="map_generate"):
                stats = state_df[metric_col].describe().to_dict()
                prompt = (
                    f"You are analyzing Indian state-level Aadhaar data. Metric: {metric_label}. "
                    f"Top state: {top_row['state']} ({fmt_value(top_row[metric_col])}). "
                    f"Bottom state: {bottom_row['state']} ({fmt_value(bottom_row[metric_col])}). "
                    f"Summary stats: {stats}. Provide 3 insights and 2 operational implications."
                )
                if custom_q:
                    prompt += f" User question: {custom_q}"
                try:
                    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    response = client.models.generate_content(
                        model=map_model,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                        config=types.GenerateContentConfig(temperature=0.3),
                    )
                    st.write(response.text)
                except Exception as exc:
                    st.error(f"Error: {exc}")

# Trends
with tabs[2]:
    st.subheader("Daily Dynamics (dense months)")
    daily_focus = load_parquet("daily_focus")
    daily = daily_focus.groupby("date", as_index=False)[["enrol_total", "demo_total", "bio_total"]].sum().sort_values("date")
    fig_daily = px.line(
        daily,
        x="date",
        y=["enrol_total", "demo_total", "bio_total"],
        labels={"value": "Transactions", "date": "Date", "variable": "Metric"},
    )
    fig_daily.update_layout(height=420)
    st.plotly_chart(fig_daily, use_container_width=True)

    st.subheader("Day-of-Week Pattern")
    fig_dow = px.bar(
        dow_summary,
        x="dow",
        y=["enrol_total", "demo_total", "bio_total"],
        barmode="group",
        labels={"value": "Average Transactions", "dow": "Day"},
    )
    fig_dow.update_layout(height=380)
    st.plotly_chart(fig_dow, use_container_width=True)

# State Deep Dive
with tabs[3]:
    st.subheader("State Deep Dive")
    state_df = state_daily.copy()
    if selected_state != "All States":
        state_df = state_df[state_df["state"] == selected_state]

    state_df = state_df.sort_values("date")
    st.write(f"Rows: {len(state_df):,}")
    fig_state = px.line(
        state_df,
        x="date",
        y=["enrol_total", "demo_total", "bio_total"],
        labels={"value": "Transactions", "date": "Date", "variable": "Metric"},
    )
    fig_state.update_layout(height=420)
    st.plotly_chart(fig_state, use_container_width=True)

    if selected_state != "All States":
        st.subheader("Monthly Summary")
        state_month = state_monthly[state_monthly["state"] == selected_state].sort_values("month")
        fig_state_month = go.Figure()
        fig_state_month.add_bar(x=state_month["month"], y=state_month["demo_total"], name="Demographic", marker_color="#ff7f0e")
        fig_state_month.add_bar(x=state_month["month"], y=state_month["bio_total"], name="Biometric", marker_color="#2ca02c")
        fig_state_month.add_trace(go.Scatter(
            x=state_month["month"], y=state_month["enrol_total"], name="Enrolment", marker_color="#1f77b4",
            mode="lines+markers",
        ))
        fig_state_month.update_layout(barmode="stack", height=380, legend_orientation="h")
        st.plotly_chart(fig_state_month, use_container_width=True)

# District Insights
with tabs[4]:
    st.subheader("District Quadrants")
    df = district_focus.copy()
    if selected_state != "All States":
        df = df[df["state"] == selected_state]
    fig_quad = px.scatter(
        df,
        x="enrol_per_1k",
        y="updates_per_1k",
        color="quadrant",
        hover_data=["state", "district"],
        labels={"enrol_per_1k": "Enrolments per 1k", "updates_per_1k": "Updates per 1k"},
    )
    fig_quad.update_layout(height=420)
    st.plotly_chart(fig_quad, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Update/Enrol Ratio Districts")
        top_ratio = district_summary.dropna(subset=["update_enrol_ratio"]).copy()
        top_ratio = top_ratio[top_ratio["enrol_total"] >= 1000]
        top_ratio = top_ratio.sort_values("update_enrol_ratio", ascending=False).head(10)
        st.dataframe(top_ratio[["state", "district", "update_enrol_ratio"]], use_container_width=True)

    with col2:
        st.subheader("Top Updates per 1k Population")
        top_u = district_summary.dropna(subset=["updates_per_1k"]).sort_values("updates_per_1k", ascending=False).head(10)
        st.dataframe(top_u[["state", "district", "updates_per_1k"]], use_container_width=True)

# Pincode Anomalies
with tabs[5]:
    st.subheader("Pincode Update/Enrolment Anomalies")
    df = pincode_anomalies.copy()
    if selected_state != "All States":
        df = df[df["state"] == selected_state]
    df = df.sort_values("ratio_z", ascending=False).head(20)
    fig_anom = px.scatter(
        df,
        x="ratio_z",
        y=df["pincode"].astype(str),
        size="update_to_enrol_ratio",
        hover_data=["state", "district", "update_to_enrol_ratio"],
        labels={"ratio_z": "z-score", "y": "Pincode"},
    )
    fig_anom.update_layout(height=420)
    st.plotly_chart(fig_anom, use_container_width=True)
    st.dataframe(df[["pincode", "state", "district", "update_to_enrol_ratio", "ratio_z"]], use_container_width=True)

# City maps
with tabs[6]:
    st.subheader("City Pincode Maps")
    city_key = selected_city.lower()
    geo = load_geojson(city_key)
    if geo is None:
        st.warning("No geojson available for this city.")
    else:
        map_df = pd.DataFrame([feat["properties"] for feat in geo["features"]])
        map_df["pincode"] = pd.to_numeric(map_df.get("pin_code", map_df.get("pincode")), errors="coerce").astype("Int64")
        if "enrol_total" not in map_df.columns:
            map_df["enrol_total"] = 0
        if "update_total" not in map_df.columns:
            map_df["update_total"] = 0
        value_col = "update_total" if metric_choice == "Updates" else "enrol_total"
        fig_map = px.choropleth(
            map_df,
            geojson=geo,
            locations="pincode",
            color=value_col,
            featureidkey="properties.pincode",
            color_continuous_scale="Viridis",
            labels={value_col: metric_choice},
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(height=600, margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)

# AI Analyst
with tabs[7]:
    st.subheader("Gemini Data Analyst")
    st.write("Ask questions about the insights, trends, and anomalies. The assistant uses the summary metrics and top tables as context.")

    model_id = st.selectbox(
        "Model",
        ["gemini-2.5-flash-lite", "gemini-1.5-flash", "gemini-2.0-flash-exp"],
        index=0,
        help="Flash-Lite has the highest free tier limits.",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "model", "content": "Hi! I can help interpret the UIDAI dataset and dashboards."}
        ]

    context = {
        "date_range": f"{summary['date_min']} to {summary['date_max']}",
        "total_enrolments": summary["total_enrolments"],
        "total_updates": summary["total_demographic_updates"] + summary["total_biometric_updates"],
        "update_ratio": summary["overall_update_ratio"],
        "top_states_updates": summary["top_states_updates"],
        "top_states_enrolment": summary["top_states_enrolment"],
        "top_district_update_ratio": summary["top_district_update_ratio"],
        "top_district_updates_per_1k": summary["top_district_updates_per_1k"],
        "daily_focus_months": summary["daily_focus_months"],
    }
    context_text = "Context summary:\n" + json.dumps(context, indent=2)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask about enrolment patterns, anomalies, or state trends..."):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("model"):
            with st.spinner("Analyzing..."):
                try:
                    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    history = [
                        types.Content(
                            role="user" if m["role"] == "user" else "model",
                            parts=[types.Part.from_text(text=m["content"])],
                        )
                        for m in st.session_state.messages
                    ]
                    history.insert(0, types.Content(role="user", parts=[types.Part.from_text(text=context_text)]))
                    response = client.models.generate_content(
                        model=model_id,
                        contents=history,
                        config=types.GenerateContentConfig(temperature=0.3),
                    )
                    reply = response.text
                    st.write(reply)
                    st.session_state.messages.append({"role": "model", "content": reply})
                except Exception as exc:
                    st.error(f"Error: {exc}")

# Downloads
with tabs[8]:
    st.subheader("Downloads")
    report_path = BASE / "reports" / "uidai_hackathon_report.pdf"
    if report_path.exists():
        with open(report_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="uidai_hackathon_report.pdf")
    else:
        st.info("Report not generated yet.")

    st.write("Static figures are available in analysis/figures_final for submission decks.")
