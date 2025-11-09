import hopsworks
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Fetch values securely
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")

# src/features/process_features.py
import hopsworks
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import math

# --- (1) Connect to Hopsworks and load raw data ---
project = hopsworks.login(api_key_value=api_key, project=project_name) 
fs = project.get_feature_store()
raw_fg = fs.get_feature_group("aqi_raw_weather_pollution", version=1)
raw_df = raw_fg.read()

# --- (2) Compute AQI ---
def compute_subindex(Cp, breakpoints):
    for (Clow, Chigh, Ilow, Ihigh) in breakpoints:
        if Clow <= Cp <= Chigh:
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (Cp - Clow) + Ilow
    return np.nan

# define your breakpoints (same as your previous code)
# --- (4) Define CPCB breakpoints ---
pm25_breakpoints = [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)]
pm10_breakpoints = [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,500,401,500)]
no2_breakpoints = [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,500,401,500)]
so2_breakpoints = [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2000,401,500)]
co_breakpoints = [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34.1,50,401,500)]
o3_breakpoints = [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,1000,401,500)]

# --- (5) Compute subindices for all pollutants ---
raw_df['pm25_sub'] = raw_df['pm2_5'].apply(lambda x: compute_subindex(x, pm25_breakpoints))
raw_df['pm10_sub'] = raw_df['pm10'].apply(lambda x: compute_subindex(x, pm10_breakpoints))
raw_df['no2_sub'] = raw_df['nitrogen_dioxide'].apply(lambda x: compute_subindex(x, no2_breakpoints))
raw_df['so2_sub'] = raw_df['sulphur_dioxide'].apply(lambda x: compute_subindex(x, so2_breakpoints))
raw_df['co_sub'] = raw_df['carbon_monoxide'].apply(lambda x: compute_subindex(x, co_breakpoints))
raw_df['o3_sub'] = raw_df['ozone'].apply(lambda x: compute_subindex(x, o3_breakpoints))

# Final AQI = max of sub-indices
raw_df['AQI'] = raw_df[['pm25_sub', 'pm10_sub', 'no2_sub', 'so2_sub', 'co_sub', 'o3_sub']].max(axis=1)

# --- (3) Feature engineering ---
df = raw_df.copy()
df = df.drop(['sulphur_dioxide', 'carbon_monoxide'], axis=1)

# extract time features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# season
def get_season(month):
    if month in [12,1,2]: return "winter"
    elif month in [3,4,5]: return "spring"
    elif month in [6,7,8]: return "summer"
    else: return "autumn"
df['season'] = df['month'].apply(get_season)
df = pd.get_dummies(df, columns=['season'], prefix='season')

# --- ensure all possible season columns exist ---
expected_season_cols = ['season_spring', 'season_summer', 'season_autumn', 'season_winter']
for col in expected_season_cols:
    if col not in df.columns:
        df[col] = 0

if 'season_autumn' in df.columns: df = df.drop(columns=['season_autumn'])

# hour cyclic encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# day of week one-hot
df = pd.get_dummies(df, columns=['day_of_week'], prefix='dow')

# scale numeric
scaler = RobustScaler()
# dropping relative humididty
numeric = ['relative_humidity_2m','pm10','pm2_5','ozone','nitrogen_dioxide']
# numeric = ['pm10','pm2_5','ozone','nitrogen_dioxide']
df[numeric] = scaler.fit_transform(df[numeric])

# select final features
final_df = df[['timestamp'] + numeric +
              ['season_spring','season_summer','season_winter','hour_sin','hour_cos'] +
              [c for c in df.columns if c.startswith('dow_')] + ['AQI']]

print(f'owais is looking for processed features df columns: {final_df.columns.tolist()}')
# --- ✅ Fix numeric dtypes ---
int_cols = ["season_spring", "season_summer", "season_winter"]
for col in int_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].astype("int64")

# Fix AQI column casing
if "AQI" in final_df.columns:
    final_df.rename(columns={"AQI": "aqi"}, inplace=True)

# --- Insert ---
# processed_fg = fs.get_or_create_feature_group(
#     name="aqi_hourly_features",
#     version=2,
#     primary_key=["timestamp"],
#     description="Processed and engineered hourly features with AQI target"
# )

# processed_fg = fs.get_feature_group(name="aqi_hourly_features", version=2)
# processed_fg.update_schema(remove_columns=["relative_humidity_2m"])
# processed_fg.insert(final_df, write_options={"wait_for_job": False})
# print("✅ Processed data inserted successfully!")

# new one
# --- Create a new feature group without relative_humidity_2m ---
processed_fg = fs.get_or_create_feature_group(
    name="aqi_hourly_features",
    version=4,  # new version
    primary_key=["timestamp"],
    description="Processed and engineered hourly features without relative_humidity_2m"
)

# --- Insert the processed dataframe ---
processed_fg.insert(final_df, write_options={"wait_for_job": False})

print("✅ Processed data inserted successfully into new feature group (v3)!")


# print('Debugging owais')
# print("Total rows:", len(raw_df))
# print("Unique timestamps:", raw_df['timestamp'].nunique())
