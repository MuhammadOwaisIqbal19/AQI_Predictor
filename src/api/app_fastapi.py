# ==============================================
# 🌐 FastAPI Backend for AQI Prediction (Hopsworks + XGBoost)
# ==============================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import hopsworks
import joblib
import os
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")


# ==============================================
# 1️⃣ Connect to Hopsworks and Load Model
# ==============================================
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"), project="AQI_Predictor10pearls")
mr = project.get_model_registry()

# Get the latest registered model
model_entry = mr.get_model("xgb_aqi_model", version=None)  # None = latest
model_dir = model_entry.download()
bundle = joblib.load(model_dir + "/model_bundle.pkl")
model = bundle["model"]
features = bundle["features"]

print("✅ Model loaded successfully from Hopsworks Registry!")

# ==============================================
# 2️⃣ Define FastAPI app and schemas
# ==============================================
app = FastAPI(
    title="🌍 AQI Prediction API",
    description="Predict Air Quality Index (AQI) using trained XGBoost model from Hopsworks",
    version="1.0.0"
)

class AQIInput(BaseModel):
    relative_humidity_2m: float
    pm10: float
    pm2_5: float
    ozone: float
    nitrogen_dioxide: float
    hour: int
    day_of_week: int
    season: str   # 'spring', 'summer', 'winter'

# ==============================================
# 3️⃣ Helper: Encode time and categorical features
# ==============================================
def encode_input(data: AQIInput):
    """Convert raw user inputs into model-ready features"""

    # Cyclic hour encoding
    hour_sin = math.sin(2 * math.pi * data.hour / 24)
    hour_cos = math.cos(2 * math.pi * data.hour / 24)

    # One-hot encode season
    season_spring = 1 if data.season == "spring" else 0
    season_summer = 1 if data.season == "summer" else 0
    season_winter = 1 if data.season == "winter" else 0

    # One-hot encode day of week
    dow_encoded = [1 if data.day_of_week == i else 0 for i in range(7)]

    # Combine all features in correct order
    input_vector = pd.DataFrame([[
        data.relative_humidity_2m, data.pm10, data.pm2_5, data.ozone, data.nitrogen_dioxide,
        season_spring, season_summer, season_winter,
        hour_sin, hour_cos,
        *dow_encoded
    ]], columns=features)

    return input_vector

# ==============================================
# 4️⃣ Root endpoint
# ==============================================
@app.get("/")
def root():
    return {
        "message": "🌍 AQI Prediction API is running!",
        "usage": "Send a POST request to /predict with the required features.",
        "example": {
            "relative_humidity_2m": 60,
            "pm10": 40,
            "pm2_5": 25,
            "ozone": 35,
            "nitrogen_dioxide": 18,
            "hour": 14,
            "day_of_week": 3,
            "season": "summer"
        }
    }

# ==============================================
# 5️⃣ Prediction endpoint (Single AQI)
# ==============================================
@app.post("/predict")
def predict_aqi(data: AQIInput):
    """Predict AQI for given input"""
    input_vector = encode_input(data)
    pred_aqi = float(model.predict(input_vector)[0])
    return {"predicted_AQI": round(pred_aqi, 2)}

# ==============================================
# 6️⃣ Forecast endpoint (Next 72 hours)
# ==============================================
@app.post("/forecast_72hr")
def forecast_72hr(data: AQIInput):
    """Generate 72-hour AQI forecast autoregressively"""

    current_features = encode_input(data).iloc[0].copy()
    current_time = datetime.now()

    predictions, timestamps = [], []

    for i in range(1, 73):
        next_aqi = model.predict(current_features.to_frame().T)[0]
        predictions.append(next_aqi)

        prediction_time = current_time + timedelta(hours=i)
        timestamps.append(prediction_time.strftime('%Y-%m-%d %H:%M:%S'))

        # Update cyclic hour + DOW
        hour = prediction_time.hour
        day_of_week = prediction_time.weekday()
        current_features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        current_features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        for d in range(7):
            current_features[f"dow_{d}"] = 1 if d == day_of_week else 0

    forecast_df = pd.DataFrame({
        "timestamp": timestamps,
        "predicted_AQI": np.round(predictions, 2)
    })

    return {
        "message": "✅ 72-hour AQI forecast generated successfully!",
        "forecast": forecast_df.to_dict(orient="records")
    }
