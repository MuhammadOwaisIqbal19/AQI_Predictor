# ==============================================
# 🎨 Streamlit Frontend for AQI Prediction (FastAPI + Plotly)
# ==============================================

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")

# ==============================================
# 1️⃣ FastAPI backend URL
# ==============================================
API_URL = "http://127.0.0.1:8000"  # Change if deployed elsewhere

st.set_page_config(page_title="AQI Prediction Dashboard", page_icon="🌍", layout="wide")

st.title("🌍 Air Quality Index (AQI) Prediction Dashboard")
st.markdown("Predict and visualize **real-time AQI** and **72-hour forecast** using ML model trained on weather & pollutant data.")

# ==============================================
# 2️⃣ User Inputs
# ==============================================
st.sidebar.header("🌦️ Input Parameters")

humidity = st.sidebar.number_input("Relative Humidity (%):", min_value=0, max_value=100, value=60)
pm10 = st.sidebar.number_input("PM10 (μg/m³):", min_value=0, max_value=500, value=40)
pm25 = st.sidebar.number_input("PM2.5 (μg/m³):", min_value=0, max_value=500, value=25)
ozone = st.sidebar.number_input("Ozone (μg/m³):", min_value=0, max_value=1000, value=35)
no2 = st.sidebar.number_input("Nitrogen Dioxide (μg/m³):", min_value=0, max_value=500, value=20)

# Derived time inputs
now = datetime.now()
hour = st.sidebar.slider("Hour of Day (0-23):", 0, 23, now.hour)
dow = now.weekday()
season = st.sidebar.selectbox("Season:", ["spring", "summer", "winter"])

# ==============================================
# 3️⃣ Prepare request data
# ==============================================
input_data = {
    "relative_humidity_2m": humidity,
    "pm10": pm10,
    "pm2_5": pm25,
    "ozone": ozone,
    "nitrogen_dioxide": no2,
    "hour": hour,
    "day_of_week": dow,
    "season": season
}

# ==============================================
# 4️⃣ Layout sections
# ==============================================
tab1, tab2 = st.tabs(["📍 Current AQI Prediction", "📈 72-Hour Forecast"])

# ==============================================
# 5️⃣ Single AQI Prediction
# ==============================================
with tab1:
    st.subheader("📍 Current AQI Prediction")
    if st.button("Predict AQI"):
        with st.spinner("Contacting model API..."):
            try:
                res = requests.post(f"{API_URL}/predict", json=input_data)
                if res.status_code == 200:
                    result = res.json()
                    st.success(f"✅ Predicted AQI: **{result['predicted_AQI']}**")
                    aqi = result['predicted_AQI']

                    # AQI Color Categories (Simplified)
                    if aqi <= 50:
                        color = "#00E400"; label = "Good"
                    elif aqi <= 100:
                        color = "#FFFF00"; label = "Moderate"
                    elif aqi <= 150:
                        color = "#FF7E00"; label = "Unhealthy (Sensitive)"
                    elif aqi <= 200:
                        color = "#FF0000"; label = "Unhealthy"
                    elif aqi <= 300:
                        color = "#99004C"; label = "Very Unhealthy"
                    else:
                        color = "#7E0023"; label = "Hazardous"

                    st.markdown(f"### 🟩 AQI Category: **{label}**")
                    st.progress(min(aqi / 500, 1.0))
                    st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:10px;color:white;text-align:center'>AQI = {aqi} ({label})</div>", unsafe_allow_html=True)
                else:
                    st.error("❌ API Error. Please ensure FastAPI server is running.")
            except Exception as e:
                st.error(f"⚠️ Connection failed: {e}")

# ==============================================
# 6️⃣ 72-Hour Forecast
# ==============================================
with tab2:
    st.subheader("📈 AQI Forecast (Next 72 Hours)")

    if st.button("Generate 72-hour Forecast"):
        with st.spinner("Fetching forecast from FastAPI backend..."):
            try:
                res = requests.post(f"{API_URL}/forecast_72hr", json=input_data)
                if res.status_code == 200:
                    forecast = res.json()["forecast"]
                    forecast_df = pd.DataFrame(forecast)
                    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])

                    st.success("✅ Forecast generated successfully!")

                    # Plot
                    fig = px.line(
                        forecast_df,
                        x="timestamp",
                        y="predicted_AQI",
                        title="72-Hour AQI Forecast",
                        labels={"predicted_AQI": "Predicted AQI", "timestamp": "Time"},
                        line_shape="spline",
                    )
                    fig.update_traces(line_color="#FF4B4B", line_width=3)
                    st.plotly_chart(fig, use_container_width=True)

                    # Data Table
                    st.dataframe(forecast_df)
                else:
                    st.error("❌ API returned an error. Check FastAPI logs.")
            except Exception as e:
                st.error(f"⚠️ Could not connect to API: {e}")
