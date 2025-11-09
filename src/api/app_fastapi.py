
# below with multiple models and 3 days prediction implemtation# ==============================================
# ==============================================================
# 🌍 FastAPI Backend for AQI Prediction (Auto-Select Best Model)
# ==============================================================
from fastapi import FastAPI
from pydantic import BaseModel
import hopsworks
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ==============================================================
# 1️⃣ Initialize App
# ==============================================================
app = FastAPI(title="AQI Prediction API", description="Predicts AQI using the best model from Hopsworks Registry")

load_dotenv()

# ==============================================================
# 2️⃣ Connect to Hopsworks
# ==============================================================
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT")
)
fs = project.get_feature_store()   # <--- define fs here
mr = project.get_model_registry()

# ==============================================================
# 3️⃣ Find the Best Model (Highest R²)
# ==============================================================
model_names = ["xgb_aqi_model", "ridge_aqi_model", "lstm_aqi_model"]

best_model_meta = None
best_model_name = None
best_model_type = None
best_r2 = -999


for name in model_names:
    try:
        # Get model entry
        model_entry = mr.get_model(name, version=None)  # latest version entry
        version_number = model_entry.version           # latest version number
        
        # Fetch version object (metrics are here)
        model_version = mr.get_model(name, version=version_number)
        metrics = getattr(model_version, "training_metrics", None)
        # Sometimes your metrics dict uses 'r2' or 'test_r2', check both
        r2_value = None
        if metrics:
            if "r2" in metrics:
                r2_value = metrics["r2"]
            elif "test_r2" in metrics:
                r2_value = metrics["test_r2"]    
        if r2_value is not None:  
             if r2_value > best_r2:
                best_r2 = r2_value
                best_model_meta = model_version
                best_model_name = name
                best_model_type = (
                    "lstm" if "lstm" in name else
                    "ridge" if "ridge" in name else
                    "xgb"
                )      
        
        else:
             print(f"⚠️ Model {name} v{version_number} has no R² metric")
    except Exception as e:
        print(f"⚠️ Could not read model {name}: {e}")

if best_model_meta is None:
    raise ValueError("❌ No model with R² found among latest versions of the 3 models!")

print(f"✅ Best model selected: {best_model_name} (v{best_model_meta.version}, R²={best_r2:.3f})")

# ==============================================================
# 4️⃣ Download and Load Best Model
# ==============================================================
model_dir = best_model_meta.download()

import os
import glob
import joblib
from tensorflow.keras.models import load_model

# Download best model from Hopsworks
model_dir = best_model_meta.download()
print("Files in model_dir:", os.listdir(model_dir))

# Determine model type and load accordingly
if best_model_type == "lstm":
    # LSTM is a Keras model (.h5)
    model_file = os.path.join(model_dir, "lstm_model.h5")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"LSTM model file not found: {model_file}")
    model = load_model(model_file)
    # LSTM features (already defined in your code)
    features = [
        'relative_humidity_2m',
        'pm10', 'pm2_5', 'ozone', 'nitrogen_dioxide',
        'season_spring', 'season_summer', 'season_winter',
        'hour_sin', 'hour_cos',
        'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6'
    ]
else:
    # XGBoost or Ridge (.pkl)
    pkl_file = os.path.join(model_dir, f"{best_model_type}_model.pkl")
    if not os.path.exists(pkl_file):
        # fallback: find any .pkl file
        pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl file found in {model_dir}")
        pkl_file = pkl_files[0]

    bundle = joblib.load(pkl_file)
    model = bundle["model"]
    features = bundle["features"]

print(f"✅ Loaded {best_model_type.upper()} model successfully!")


# ==============================================================
# 5️⃣ Define Input Schema
# ==============================================================
class AQIRequest(BaseModel):
    relative_humidity_2m: float
    pm10: float
    pm2_5: float
    ozone: float
    nitrogen_dioxide: float
    hour: int
    day_of_week: int
    season: str
# class AQIForecastRequest(BaseModel):
#     pm10: float
#     pm2_5: float
#     ozone: float
#     nitrogen_dioxide: float
#     hour: int
#     day_of_week: int
#     season: str

# ==============================================================
# 6️⃣ Helper: Encode Input Features
# ==============================================================
def preprocess_input(request: AQIRequest):
    data = {
        "relative_humidity_2m": request.relative_humidity_2m,
        "pm10": request.pm10,
        "pm2_5": request.pm2_5,
        "ozone": request.ozone,
        "nitrogen_dioxide": request.nitrogen_dioxide,
    }

    # Season one-hot
    for s in ["spring", "summer", "winter"]:
        data[f"season_{s}"] = 1 if request.season == s else 0

    # Hour sin/cos
    data["hour_sin"] = np.sin(2 * np.pi * request.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * request.hour / 24)

    # Day of week one-hot
    for d in range(7):
        data[f"dow_{d}"] = 1 if d == request.day_of_week else 0

    return data

# ==============================================================
# 7️⃣ Root Endpoint
# ==============================================================
@app.get("/")
def root():
    return {
        "message": "🌍 AQI Prediction API",
        "best_model_used": best_model_name,
        "model_version": best_model_meta.version,
        "best_r2": best_r2,
        "endpoints": {
            "/predict": "Predict single AQI value",
            "/forecast_3day": "Predict next 3 days AQI (1 per day)"
        }
    }


from fastapi import APIRouter
import numpy as np
import pandas as pd

@app.get("/predict")  # <-- only GET is needed
def predict_today():
    """
    Predict today's AQI using the latest processed record
    from the Hopsworks feature store.
    """
    try:
        # ✅ Read last record from the latest processed data
        fg = fs.get_feature_group("aqi_hourly_features", version=4)
        df = fg.read().sort_values("timestamp").reset_index(drop=True)
        last_row = df.iloc[[-1]].copy()

        # ✅ Keep only valid model features
        valid_features = [f for f in features if f in last_row.columns]
        X = last_row[valid_features]

        # ✅ Predict based on model type
        if best_model_type == "lstm":
            X_scaled = lstm_scaler.transform(X)
            X_reshaped = np.expand_dims(X_scaled, axis=0)
            pred = model.predict(X_reshaped)[0][0]
        else:
            pred = model.predict(X)[0]

        # ✅ Return structured response
        return {
            "predicted_AQI": float(pred),
            "date": str(last_row["timestamp"].values[0]),
            "model_used": best_model_name,
            "r2": best_r2
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================================== 
# Forecast Next 3 Days AQI (Autoregressive)
# ============================================================== 
@app.get("/forecast_3day")
def forecast_next_3_days_autoregressive():
    # Load last row of processed data as base
    fg = fs.get_feature_group("aqi_hourly_features", version=4)
    df = fg.read().sort_values("timestamp").reset_index(drop=True)
    last_row = df.iloc[[-1]].copy()  # DataFrame
    timestamp = last_row['timestamp'].values[0]
    now = pd.to_datetime(timestamp)

    # Keep only model features
    # last_row = last_row[[f for f in features if f in last_row.columns]]

    forecasts = []
    current_input = last_row.copy()
    now = pd.to_datetime(last_row['timestamp'].values[0])

    for i in range(1, 4):  # next 3 days
        future = now + pd.Timedelta(days=i)
        fdict = current_input.iloc[0].to_dict()

        # Update temporal features
        fdict["hour_sin"] = np.sin(2 * np.pi * future.hour / 24)
        fdict["hour_cos"] = np.cos(2 * np.pi * future.hour / 24)
        dow = future.weekday()
        for d in range(7):
            fdict[f"dow_{d}"] = 1 if d == dow else 0
        month = future.month
        fdict["season_spring"] = 1 if month in [3, 4, 5] else 0
        fdict["season_summer"] = 1 if month in [6, 7, 8] else 0
        fdict["season_winter"] = 1 if month in [12, 1, 2] else 0

        df_future = pd.DataFrame([fdict])
        df_future = df_future[[f for f in features if f in df_future.columns]]

        if best_model_type == "lstm":
            X_scaled = lstm_scaler.transform(df_future)
            X_reshaped = np.expand_dims(X_scaled, axis=0)
            pred = model.predict(X_reshaped)[0][0]
        else:
            pred = model.predict(df_future)[0]

        # Store forecast
        forecasts.append({
            "day": f"Day {i}",
            "date": future.strftime("%Y-%m-%d"),
            "predicted_AQI": float(pred)
        })


    return {
        "forecast_next_3_days_autoregressive": forecasts,
        "model_used": best_model_name,
        "model_version": best_model_meta.version,
        "best_r2": best_r2
    }