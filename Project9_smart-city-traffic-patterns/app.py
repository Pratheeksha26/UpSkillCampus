# app.py
"""Streamlit dashboard for Smart City Traffic Patterns.
Features:
1️⃣ Exploratory Data Analysis (EDA)
2️⃣ Forecasting with Prophet (trained on full data, optional rolling window)
3️⃣ Infrastructure Planning Insights
"""

import streamlit as st
import pandas as pd
from utils import load_data, add_holiday_flag, generate_eda_plots, train_forecast_model, get_insights, download_forecast_csv

st.set_page_config(page_title="Smart City Traffic Dashboard", layout="wide")

st.title("🚦 Smart City Traffic Dashboard")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
junction = st.sidebar.selectbox("Select Junction", [1, 2, 3, 4], index=0)
show_holidays = st.sidebar.checkbox("Treat weekends as holidays", value=True)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 1, 30, 7)
use_rolling = st.sidebar.checkbox("Use rolling 6‑month window for training", value=False)

@st.cache_data(show_spinner=False)
def get_data():
    df = load_data()
    df = add_holiday_flag(df, treat_weekends=show_holidays)
    return df

data = get_data()

junction_df = data[data["Junction"] == junction].copy()

# ---- EDA ----
st.header("📊 Exploratory Data Analysis")
eda_tabs = st.tabs(["Time Series", "Distribution", "Heatmap"])
with eda_tabs[0]:
    st.plotly_chart(generate_eda_plots.time_series(junction_df), use_container_width=True)
with eda_tabs[1]:
    st.plotly_chart(generate_eda_plots.distribution(junction_df), use_container_width=True)
with eda_tabs[2]:
    st.plotly_chart(generate_eda_plots.heatmap(junction_df), use_container_width=True)

# ---- Forecast ----
st.header("🔮 Traffic Forecast")
model_output = train_forecast_model(junction_df, horizon_days=forecast_horizon, rolling=use_rolling)
st.plotly_chart(model_output["forecast_plot"], use_container_width=True)

# Download button
csv_bytes = download_forecast_csv(model_output["forecast"])
st.download_button("📥 Download Forecast CSV", data=csv_bytes, file_name=f"forecast_junction_{junction}.csv", mime="text/csv")

# ---- Insights ----
st.header("🛠️ Infrastructure Planning Insights")
insights = get_insights(junction_df, model_output["forecast"])
for key, value in insights.items():
    st.metric(label=key, value=value)

st.caption("*Insights are based on peak hour vehicle counts and forecasted peaks. Adjust settings for different horizons or training windows.*")
