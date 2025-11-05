import hopsworks
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")

import pandas as pd
from openmeteo_requests import Client
import requests_cache
from retry_requests import retry

# ============ FETCH RAW WEATHER + POLLUTION DATA ============
def fetch_raw_data():
    LAT, LON = 31.558, 74.3507
    START, END = "2024-05-01", "2025-11-03"
    TIMEZONE = "Asia/Karachi"

    weather_vars = ["relative_humidity_2m"]
    pollutant_vars = ["pm10", "pm2_5", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]

    cache_session = requests_cache.CachedSession(".cache_all", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = Client(session=retry_session)

    # Weather data
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": LAT, "longitude": LON,
        "start_date": START, "end_date": END,
        "hourly": weather_vars, "timezone": TIMEZONE
    }
    weather_res = client.weather_api(weather_url, params=weather_params)[0]
    w_hourly = weather_res.Hourly()
    w_values = [w_hourly.Variables(i).ValuesAsNumpy() for i in range(len(weather_vars))]
    w_times = pd.date_range(
        start=pd.to_datetime(w_hourly.Time(), unit="s", utc=True).tz_convert(TIMEZONE),
        periods=len(w_values[0]), freq=pd.Timedelta(seconds=w_hourly.Interval()), inclusive="left"
    )
    weather_df = pd.DataFrame({weather_vars[i]: w_values[i] for i in range(len(weather_vars))})
    weather_df["timestamp"] = w_times
    weather_df = weather_df[["timestamp"] + weather_vars]

    # Pollution data
    pollution_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    pollution_params = {
        "latitude": LAT, "longitude": LON,
        "start_date": START, "end_date": END,
        "hourly": pollutant_vars, "timezone": TIMEZONE
    }
    pollution_res = client.weather_api(pollution_url, params=pollution_params)[0]
    p_hourly = pollution_res.Hourly()
    p_values = [p_hourly.Variables(i).ValuesAsNumpy() for i in range(len(pollutant_vars))]
    p_times = pd.date_range(
        start=pd.to_datetime(p_hourly.Time(), unit="s", utc=True).tz_convert(TIMEZONE),
        periods=len(p_values[0]), freq=pd.Timedelta(seconds=p_hourly.Interval()), inclusive="left"
    )
    pollution_df = pd.DataFrame({pollutant_vars[i]: p_values[i] for i in range(len(pollutant_vars))})
    pollution_df["timestamp"] = p_times
    pollution_df = pollution_df[["timestamp"] + pollutant_vars]

    merged_df = pd.merge(weather_df, pollution_df, on="timestamp", how="inner")
    return merged_df


# ============ PUSH TO HOPSWORKS RAW FEATURE GROUP ============
def push_to_hopsworks(merged_df):
    project = hopsworks.login(api_key_value=api_key, project=project_name)  
    fs = project.get_feature_store()
    raw_fg = fs.get_or_create_feature_group(
        name="aqi_raw_weather_pollution",
        version=1,
        primary_key=["timestamp"],
        description="Raw hourly weather and pollution data fetched from OpenMeteo APIs"
    )
    raw_fg.insert(merged_df, write_options={"wait_for_job": False})
    print("✅ Raw data inserted to Hopsworks Feature Group!")




if __name__ == "__main__":
    df = fetch_raw_data()
    push_to_hopsworks(df)