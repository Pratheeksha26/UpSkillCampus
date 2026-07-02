# utils.py
"""Utility helper functions for the Smart City Traffic Streamlit dashboard.
All heavy‐lifting (data loading, preprocessing, EDA plot creation, forecasting,
and insight extraction) lives here so that ``app.py`` stays tidy.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from io import BytesIO

DATA_PATH = "d:/Internship/UCT/Project4_Ag_Prediction of Agriculture Crop Production In India/Project9_smart-city-traffic-patterns/smart-city-traffic-patterns/train_aWnotuB.csv"

# ------------------------------------------------------------
# Data loading & basic preprocessing
# ------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Read the raw CSV, parse dates, and cast appropriate dtypes.
    Returns a DataFrame with columns: DateTime, Junction, Vehicles, ID.
    """
    df = pd.read_csv(DATA_PATH)
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y-%m-%d %H:%M:%S")
    # Ensure numeric types
    df["Junction"] = df["Junction"].astype(int)
    df["Vehicles"] = df["Vehicles"].astype(int)
    return df

def add_holiday_flag(df: pd.DataFrame, treat_weekends: bool = True) -> pd.DataFrame:
    """Add a boolean ``is_holiday`` column.
    If ``treat_weekends`` is True, Saturdays and Sundays are marked as holidays.
    Otherwise the column defaults to ``False`` – the user can later replace it with a
    custom holiday list.
    """
    df = df.copy()
    if treat_weekends:
        df["is_holiday"] = df["DateTime"].dt.weekday.isin([5, 6])
    else:
        df["is_holiday"] = False
    return df

# ------------------------------------------------------------
# EDA visualisations – returned as Plotly Figure objects
# ------------------------------------------------------------
class generate_eda_plots:
    @staticmethod
    def time_series(df: pd.DataFrame) -> go.Figure:
        fig = px.line(df, x="DateTime", y="Vehicles", title="Hourly Vehicle Count")
        fig.update_layout(xaxis_title="Date & Time", yaxis_title="Vehicles")
        return fig

    @staticmethod
    def distribution(df: pd.DataFrame) -> go.Figure:
        fig = px.histogram(df, x="Vehicles", nbins=30, title="Vehicle Count Distribution")
        fig.update_layout(xaxis_title="Vehicles", yaxis_title="Frequency")
        return fig

    @staticmethod
    def heatmap(df: pd.DataFrame) -> go.Figure:
        # Pivot to hours vs. days of week
        df = df.copy()
        df["hour"] = df["DateTime"].dt.hour
        df["dow"] = df["DateTime"].dt.dayofweek
        pivot = df.pivot_table(values="Vehicles", index="dow", columns="hour", aggfunc="mean")
        fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]))
        fig.update_layout(title="Average Vehicles – Hour vs. Day of Week", xaxis_title="Hour of Day", yaxis_title="Day of Week")
        return fig

# ------------------------------------------------------------
# Forecasting utilities – Prophet model per junction
# ------------------------------------------------------------
def train_forecast_model(df: pd.DataFrame, horizon_days: int = 7, rolling: bool = False) -> dict:
    """Train a Prophet model and return a dictionary with:
    * ``forecast`` – DataFrame of future predictions.
    * ``forecast_plot`` – Plotly figure showing history + forecast.
    ``rolling`` decides whether to train on the last 6‑months of data (best effort).
    """
    model_df = df.copy()
    # Prophet expects columns ds (date) and y (target)
    model_df = model_df.rename(columns={"DateTime": "ds", "Vehicles": "y"})
    if rolling:
        cutoff = model_df["ds"].max() - pd.DateOffset(months=6)
        model_df = model_df[model_df["ds"] >= cutoff]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    m.fit(model_df)
    future = m.make_future_dataframe(periods=horizon_days * 24, freq="H")
    forecast = m.predict(future)
    # Plotly visualisation
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model_df["ds"], y=model_df["y"], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper CI", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower CI", line=dict(dash="dash")))
    fig.update_layout(title="Traffic Forecast (Prophet)", xaxis_title="DateTime", yaxis_title="Vehicles")
    return {"forecast": forecast, "forecast_plot": fig}

# ------------------------------------------------------------
# Insight generation
# ------------------------------------------------------------
def get_insights(df: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """Derive simple infrastructure insights.
    Returns a mapping suitable for ``st.metric``.
    """
    # Peak hour in historical data
    peak_hour = df.groupby(df["DateTime"].dt.hour)["Vehicles"].mean().idxmax()
    # Forecasted peak within the horizon
    future_peak = forecast.loc[forecast["yhat"].idxmax()]
    insights = {
        "Peak Hour (hist)": f"{peak_hour}:00",
        "Forecasted Peak Vehicles": int(future_peak["yhat"].round()),
        "Forecasted Peak Time": str(future_peak["ds"]).split(" ")[0] + " " + str(future_peak["ds"]).split(" ")[1][:5],
        "Recommended Capacity": f"{int(future_peak["yhat"].round() * 1.2)} vehicles/hr (20% buffer)"
    }
    return insights

def download_forecast_csv(forecast_df: pd.DataFrame) -> bytes:
    """Return CSV bytes for Streamlit ``download_button``.
    Only the essential columns (ds, yhat, yhat_lower, yhat_upper) are exported.
    """
    subset = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    subset.rename(columns={"ds": "DateTime", "yhat": "Forecast", "yhat_lower": "Lower_CI", "yhat_upper": "Upper_CI"}, inplace=True)
    csv_buffer = BytesIO()
    subset.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

# End of utils.py
