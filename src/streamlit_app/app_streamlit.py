
# code with 3 multiple models and 3 days prediction

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# ==============================================
# 🌍 AQI Prediction Dashboard
# ==============================================
API_URL = "https://aqi-fastapi-backend.onrender.com"  # Update if running locally
st.set_page_config(page_title="AQI Prediction Dashboard", page_icon="🌍", layout="wide")

st.title("🌍 Air Quality Index (AQI) Prediction Dashboard")
st.markdown("""
This dashboard predicts **real-time AQI** and **3-day forecasts**  
powered by **XGBoost, Ridge Regression, or LSTM** — whichever performs best (highest R²) from the Hopsworks Model Registry.
""")

# ==============================================
# 🌦️ Sidebar Inputs
# ==============================================
st.sidebar.header("🌦️ Input Parameters")

humidity = st.sidebar.slider("Relative Humidity (%)", 0, 100, 60)
pm10 = st.sidebar.number_input("PM10 (μg/m³)", 0, 500, 40)
pm25 = st.sidebar.number_input("PM2.5 (μg/m³)", 0, 500, 25)
ozone = st.sidebar.number_input("Ozone (μg/m³)", 0, 1000, 35)
no2 = st.sidebar.number_input("Nitrogen Dioxide (μg/m³)", 0, 500, 20)
hour = datetime.now().hour
dow = datetime.now().weekday()
season = st.sidebar.selectbox("Season", ["spring", "summer", "winter"])

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

tab1, tab2 = st.tabs(["📍 Current AQI", "📈 3-Day Forecast"])

# ==============================================
# 📍 Single AQI Prediction
# ==============================================
# with tab1:
#     st.subheader("📍 Current AQI Prediction")

#     if st.button("Predict AQI"):
#         try:
#             res = requests.post(f"{API_URL}/predict", json=input_data)
#             if res.status_code == 200:
#                 result = res.json()
#                 aqi = result["predicted_AQI"]
#                 model_used = result.get("model_used", "unknown")
#                 r2 = result.get("r2", None)

#                 st.success(f"Predicted AQI: **{aqi:.2f}**")
#                 if r2 is not None:
#                     st.info(f"🧠 Model Used: `{model_used}` | R² = {r2:.3f}")
#                 else:
#                     st.info(f"🧠 Model Used: `{model_used}` | R² = N/A")
    
#                 # st.info(f"🧠 Model Used: `{model_used}` | R² = {r2:.3f}")

#                 # AQI category visualization
#                 if aqi <= 50:
#                     color, label = "#00E400", "Good"
#                 elif aqi <= 100:
#                     color, label = "#FFFF00", "Moderate"
#                 elif aqi <= 150:
#                     color, label = "#FF7E00", "Unhealthy (Sensitive)"
#                 elif aqi <= 200:
#                     color, label = "#FF0000", "Unhealthy"
#                 elif aqi <= 300:
#                     color, label = "#99004C", "Very Unhealthy"
#                 else:
#                     color, label = "#7E0023", "Hazardous"

#                 st.markdown(
#                     f"<div style='background-color:{color};padding:15px;border-radius:10px;"
#                     f"text-align:center;color:white;font-size:20px;'>AQI = {aqi:.2f} ({label})</div>",
#                     unsafe_allow_html=True
#                 )
#             else:
#                 st.error(f"❌ API Error: {res.text}")
#         except Exception as e:
#             st.error(f"⚠️ Could not connect to API: {e}")

with tab1:
    st.subheader("📍 Current AQI Prediction")

    if st.button("Current AQI"):
        try:
            # ✅ Direct GET request since /predict uses last training data row
            res = requests.get(f"{API_URL}/predict")

            if res.status_code == 200:
                result = res.json()
                aqi = result["predicted_AQI"]
                model_used = result.get("model_used", "unknown")
                r2 = result.get("r2", None)
                date = result.get("date", "Today")

                st.success(f"Predicted AQI for {date}: **{aqi:.2f}**")
                if r2 is not None:
                    st.info(f"🧠 Model Used: `{model_used}` | R² = {r2:.3f}")
                else:
                    st.info(f"🧠 Model Used: `{model_used}` | R² = N/A")

                # AQI Category Visualization
                if aqi <= 50:
                    color, label = "#00E400", "Good"
                elif aqi <= 100:
                    color, label = "#FFFF00", "Moderate"
                elif aqi <= 150:
                    color, label = "#FF7E00", "Unhealthy (Sensitive)"
                elif aqi <= 200:
                    color, label = "#FF0000", "Unhealthy"
                elif aqi <= 300:
                    color, label = "#99004C", "Very Unhealthy"
                else:
                    color, label = "#7E0023", "Hazardous"

                st.markdown(
                    f"<div style='background-color:{color};padding:15px;border-radius:10px;"
                    f"text-align:center;color:white;font-size:20px;'>AQI = {aqi:.2f} ({label})</div>",
                    unsafe_allow_html=True
                )

            else:
                st.error(f"❌ API Error: {res.text}")

        except Exception as e:
            st.error(f"⚠️ Could not connect to API: {e}")

# ==============================================
# 📈 3-Day Forecast
# ==============================================
with tab2:
    st.subheader("📈 3-Day AQI Forecast")

    if st.button("Generate 3-Day Forecast"):
        try:
            # Old (wrong)
            # response = requests.post(f"{API_URL}/forecast_3day", json=payload)
            res = requests.post(f"{API_URL}/forecast_3day", json=input_data)
            if res.status_code == 200:
                data = res.json()
                forecast = data["forecast"]
                model_used = data.get("model_used", "unknown")
                version = data.get("model_version", "N/A")
                r2 = data.get("best_r2", None)

                df = pd.DataFrame(forecast)
                # st.success(f"✅ Forecast generated successfully using `{model_used}` (v{version}) | R² = {r2:.3f}")
                if r2 is not None:
                    st.success(f"🧠 Forecast generated successfully using: `{model_used}` (v{version}) | R² = {r2:.3f}")
                else:
                    st.success(f"🧠 Forecast generated successfully using: `{model_used}` (v{version}) | R² = N/A")    
                # Use correct column name (check your actual JSON keys)
                #df ko forecast_df kia hhai
        
                x_col = 'timestamp' if 'timestamp' in df.columns else 'date'

                # fig = px.bar(df, x="date", y="predicted_AQI", color="predicted_AQI",
                #              color_continuous_scale="YlOrRd",
                #              title="Predicted AQI for Next 3 Days")
                fig = px.bar(df, x=x_col, y="predicted_AQI", color="predicted_AQI",
                             color_continuous_scale="YlOrRd",
                             title="Predicted AQI for Next 3 Days")
                fig.update_layout(xaxis_title="Date", yaxis_title="Predicted AQI", title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(df)
            else:
                st.error(f"❌ API Error: {res.text}")
        except Exception as e:
            st.error(f"⚠️ Connection error: {e}")
            # print(e)
            # error_msg = str(e) if e is not None else "Unknown error"
            # st.error(f"⚠️ Connection error: {error_msg}")
            st.stop()
                
