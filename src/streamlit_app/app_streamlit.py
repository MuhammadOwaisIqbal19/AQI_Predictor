# # ==============================================
# # 🎨 Streamlit Frontend for AQI Prediction (FastAPI + Plotly)
# # ==============================================

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px
# from datetime import datetime
# from dotenv import load_dotenv
# import os
# load_dotenv()
# api_key = os.getenv("HOPSWORKS_API_KEY")
# project_name = os.getenv("HOPSWORKS_PROJECT")

# # ==============================================
# # 1️⃣ FastAPI backend URL
# # ==============================================
# API_URL = "http://127.0.0.1:8000"  # Change if deployed elsewhere

# st.set_page_config(page_title="AQI Prediction Dashboard", page_icon="🌍", layout="wide")

# st.title("🌍 Air Quality Index (AQI) Prediction Dashboard")
# st.markdown("Predict and visualize **real-time AQI** and **72-hour forecast** using ML model trained on weather & pollutant data.")

# # ==============================================
# # 2️⃣ User Inputs
# # ==============================================
# st.sidebar.header("🌦️ Input Parameters")

# humidity = st.sidebar.number_input("Relative Humidity (%):", min_value=0, max_value=100, value=60)
# pm10 = st.sidebar.number_input("PM10 (μg/m³):", min_value=0, max_value=500, value=40)
# pm25 = st.sidebar.number_input("PM2.5 (μg/m³):", min_value=0, max_value=500, value=25)
# ozone = st.sidebar.number_input("Ozone (μg/m³):", min_value=0, max_value=1000, value=35)
# no2 = st.sidebar.number_input("Nitrogen Dioxide (μg/m³):", min_value=0, max_value=500, value=20)

# # Derived time inputs
# now = datetime.now()
# hour = st.sidebar.slider("Hour of Day (0-23):", 0, 23, now.hour)
# dow = now.weekday()
# season = st.sidebar.selectbox("Season:", ["spring", "summer", "winter"])

# # ==============================================
# # 3️⃣ Prepare request data
# # ==============================================
# input_data = {
#     "relative_humidity_2m": humidity,
#     "pm10": pm10,
#     "pm2_5": pm25,
#     "ozone": ozone,
#     "nitrogen_dioxide": no2,
#     "hour": hour,
#     "day_of_week": dow,
#     "season": season
# }

# # ==============================================
# # 4️⃣ Layout sections
# # ==============================================
# tab1, tab2 = st.tabs(["📍 Current AQI Prediction", "📈 72-Hour Forecast"])

# # ==============================================
# # 5️⃣ Single AQI Prediction
# # ==============================================
# with tab1:
#     st.subheader("📍 Current AQI Prediction")
#     if st.button("Predict AQI"):
#         with st.spinner("Contacting model API..."):
#             try:
#                 res = requests.post(f"{API_URL}/predict", json=input_data)
#                 if res.status_code == 200:
#                     result = res.json()
#                     st.success(f"✅ Predicted AQI: **{result['predicted_AQI']}**")
#                     aqi = result['predicted_AQI']

#                     # AQI Color Categories (Simplified)
#                     if aqi <= 50:
#                         color = "#00E400"; label = "Good"
#                     elif aqi <= 100:
#                         color = "#FFFF00"; label = "Moderate"
#                     elif aqi <= 150:
#                         color = "#FF7E00"; label = "Unhealthy (Sensitive)"
#                     elif aqi <= 200:
#                         color = "#FF0000"; label = "Unhealthy"
#                     elif aqi <= 300:
#                         color = "#99004C"; label = "Very Unhealthy"
#                     else:
#                         color = "#7E0023"; label = "Hazardous"

#                     st.markdown(f"### 🟩 AQI Category: **{label}**")
#                     st.progress(min(aqi / 500, 1.0))
#                     st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:10px;color:white;text-align:center'>AQI = {aqi} ({label})</div>", unsafe_allow_html=True)
#                 else:
#                     st.error("❌ API Error. Please ensure FastAPI server is running.")
#             except Exception as e:
#                 st.error(f"⚠️ Connection failed: {e}")

# # ==============================================
# # 6️⃣ 72-Hour Forecast
# # ==============================================
# with tab2:
#     st.subheader("📈 AQI Forecast (Next 72 Hours)")

#     if st.button("Generate 72-hour Forecast"):
#         with st.spinner("Fetching forecast from FastAPI backend..."):
#             try:
#                 res = requests.post(f"{API_URL}/forecast_72hr", json=input_data)
#                 if res.status_code == 200:
#                     forecast = res.json()["forecast"]
#                     forecast_df = pd.DataFrame(forecast)
#                     forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])

#                     st.success("✅ Forecast generated successfully!")

#                     # Plot
#                     fig = px.line(
#                         forecast_df,
#                         x="timestamp",
#                         y="predicted_AQI",
#                         title="72-Hour AQI Forecast",
#                         labels={"predicted_AQI": "Predicted AQI", "timestamp": "Time"},
#                         line_shape="spline",
#                     )
#                     fig.update_traces(line_color="#FF4B4B", line_width=3)
#                     st.plotly_chart(fig, use_container_width=True)

#                     # Data Table
#                     st.dataframe(forecast_df)
#                 else:
#                     st.error("❌ API returned an error. Check FastAPI logs.")
#             except Exception as e:
#                 st.error(f"⚠️ Could not connect to API: {e}")


# #-------------------------new code version below----------------------------#

# # ==============================================
# # 🎨 Streamlit Frontend for AQI Prediction (FastAPI + Plotly)
# # ==============================================

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px
# from datetime import datetime

# # ==============================================
# # 1️⃣ Backend API URL
# # ==============================================
# # API_URL = "http://127.0.0.1:8000"  # Replace with deployed FastAPI URL if hosted
# API_URL = "https://aqi-fastapi-backend.onrender.com"

# st.set_page_config(page_title="AQI Prediction Dashboard", page_icon="🌍", layout="wide")

# st.title("🌍 Air Quality Index (AQI) Prediction Dashboard")
# st.markdown("Predict and visualize **real-time AQI** and **72-hour forecasts** powered by XGBoost + Hopsworks Model Registry.")

# # ==============================================
# # 2️⃣ Sidebar Inputs
# # ==============================================
# st.sidebar.header("🌦️ Input Parameters")

# # humidity = st.sidebar.slider("Relative Humidity (%)", 0, 100, 60)
# pm10 = st.sidebar.number_input("PM10 (μg/m³)", 0, 500, 40)
# pm25 = st.sidebar.number_input("PM2.5 (μg/m³)", 0, 500, 25)
# ozone = st.sidebar.number_input("Ozone (μg/m³)", 0, 1000, 35)
# no2 = st.sidebar.number_input("Nitrogen Dioxide (μg/m³)", 0, 500, 20)
# hour = datetime.now().hour
# dow = datetime.now().weekday()
# season = st.sidebar.selectbox("Season", ["spring", "summer", "winter"])

# input_data = {
#     # "relative_humidity_2m": humidity,
#     "pm10": pm10,
#     "pm2_5": pm25,
#     "ozone": ozone,
#     "nitrogen_dioxide": no2,
#     "hour": hour,
#     "day_of_week": dow,
#     "season": season
# }

# tab1, tab2 = st.tabs(["📍 Current AQI", "📈 72-Hour Forecast"])

# # ==============================================
# # 3️⃣ Single AQI Prediction
# # ==============================================
# with tab1:
#     st.subheader("📍 Current AQI Prediction")
#     if st.button("Predict AQI"):
#         try:
#             res = requests.post(f"{API_URL}/predict", json=input_data)
#             if res.status_code == 200:
#                 result = res.json()
#                 aqi = result["predicted_AQI"]
#                 st.success(f"Predicted AQI: **{aqi}**")

#                 # Category visualization
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

#                 st.markdown(f"<div style='background-color:{color};padding:15px;border-radius:10px;text-align:center;color:white;font-size:20px;'>AQI = {aqi} ({label})</div>", unsafe_allow_html=True)
#             else:
#                 st.error("❌ API returned error.")
#         except Exception as e:
#             st.error(f"⚠️ Could not connect to API: {e}")

# # ==============================================
# # 4️⃣ 72-Hour Forecast
# # ==============================================
# with tab2:
#     st.subheader("📈 72-Hour AQI Forecast")
#     if st.button("Generate Forecast"):
#         try:
#             res = requests.post(f"{API_URL}/forecast_72hr", json=input_data)
#             if res.status_code == 200:
#                 forecast = res.json()["forecast"]
#                 df = pd.DataFrame(forecast)
#                 df["timestamp"] = pd.to_datetime(df["timestamp"])

#                 st.success("✅ Forecast generated successfully!")

#                 fig = px.line(df, x="timestamp", y="predicted_AQI", title="72-Hour AQI Forecast", markers=True)
#                 fig.update_layout(xaxis_title="Time", yaxis_title="Predicted AQI", title_x=0.5)
#                 st.plotly_chart(fig, use_container_width=True)
#                 st.dataframe(df)
#             else:
#                 st.error("❌ API Error: Could not fetch forecast.")
#         except Exception as e:
#             st.error(f"⚠️ Connection error: {e}")

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
with tab1:
    st.subheader("📍 Current AQI Prediction")

    if st.button("Predict AQI"):
        try:
            res = requests.post(f"{API_URL}/predict", json=input_data)
            if res.status_code == 200:
                result = res.json()
                aqi = result["predicted_AQI"]
                model_used = result.get("model_used", "unknown")
                r2 = result.get("r2", None)

                st.success(f"Predicted AQI: **{aqi:.2f}**")
                if r2 is not None:
                    st.info(f"🧠 Model Used: `{model_used}` | R² = {r2:.3f}")
                else:
                    st.info(f"🧠 Model Used: `{model_used}` | R² = N/A")
    
                # st.info(f"🧠 Model Used: `{model_used}` | R² = {r2:.3f}")

                # AQI category visualization
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
            res = requests.post(f"{API_URL}/forecast_72hr", json=input_data)
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
        
                x_col = 'timestamp' if 'timestamp' in forecast_df.columns else 'date'
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
                
