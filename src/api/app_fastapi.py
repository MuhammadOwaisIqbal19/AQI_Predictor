# # ==============================================
# # 🌐 FastAPI Backend for AQI Prediction (Hopsworks + XGBoost)
# # ==============================================

# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import numpy as np
# import hopsworks
# import joblib
# import os
# import math
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
# import os
# load_dotenv()
# api_key = os.getenv("HOPSWORKS_API_KEY")
# project_name = os.getenv("HOPSWORKS_PROJECT")


# # ==============================================
# # 1️⃣ Connect to Hopsworks and Load Model
# # ==============================================
# project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"), project="AQI_Predictor10pearls")
# mr = project.get_model_registry()

# # Get the latest registered model
# model_entry = mr.get_model("xgb_aqi_model", version=None)  # None = latest
# model_dir = model_entry.download()
# bundle = joblib.load(model_dir + "/model_bundle.pkl")
# model = bundle["model"]
# features = bundle["features"]

# print("✅ Model loaded successfully from Hopsworks Registry!")

# # ==============================================
# # 2️⃣ Define FastAPI app and schemas
# # ==============================================
# app = FastAPI(
#     title="🌍 AQI Prediction API",
#     description="Predict Air Quality Index (AQI) using trained XGBoost model from Hopsworks",
#     version="1.0.0"
# )

# class AQIInput(BaseModel):
#     relative_humidity_2m: float
#     pm10: float
#     pm2_5: float
#     ozone: float
#     nitrogen_dioxide: float
#     hour: int
#     day_of_week: int
#     season: str   # 'spring', 'summer', 'winter'

# # ==============================================
# # 3️⃣ Helper: Encode time and categorical features
# # ==============================================
# def encode_input(data: AQIInput):
#     """Convert raw user inputs into model-ready features"""

#     # Cyclic hour encoding
#     hour_sin = math.sin(2 * math.pi * data.hour / 24)
#     hour_cos = math.cos(2 * math.pi * data.hour / 24)

#     # One-hot encode season
#     season_spring = 1 if data.season == "spring" else 0
#     season_summer = 1 if data.season == "summer" else 0
#     season_winter = 1 if data.season == "winter" else 0

#     # One-hot encode day of week
#     dow_encoded = [1 if data.day_of_week == i else 0 for i in range(7)]

#     # Combine all features in correct order
#     input_vector = pd.DataFrame([[
#         data.relative_humidity_2m, data.pm10, data.pm2_5, data.ozone, data.nitrogen_dioxide,
#         season_spring, season_summer, season_winter,
#         hour_sin, hour_cos,
#         *dow_encoded
#     ]], columns=features)

#     return input_vector

# # ==============================================
# # 4️⃣ Root endpoint
# # ==============================================
# @app.get("/")
# def root():
#     return {
#         "message": "🌍 AQI Prediction API is running!",
#         "usage": "Send a POST request to /predict with the required features.",
#         "example": {
#             "relative_humidity_2m": 60,
#             "pm10": 40,
#             "pm2_5": 25,
#             "ozone": 35,
#             "nitrogen_dioxide": 18,
#             "hour": 14,
#             "day_of_week": 3,
#             "season": "summer"
#         }
#     }

# # ==============================================
# # 5️⃣ Prediction endpoint (Single AQI)
# # ==============================================
# @app.post("/predict")
# def predict_aqi(data: AQIInput):
#     """Predict AQI for given input"""
#     input_vector = encode_input(data)
#     pred_aqi = float(model.predict(input_vector)[0])
#     return {"predicted_AQI": round(pred_aqi, 2)}

# # ==============================================
# # 6️⃣ Forecast endpoint (Next 72 hours)
# # ==============================================
# @app.post("/forecast_72hr")
# def forecast_72hr(data: AQIInput):
#     """Generate 72-hour AQI forecast autoregressively"""

#     current_features = encode_input(data).iloc[0].copy()
#     current_time = datetime.now()

#     predictions, timestamps = [], []

#     for i in range(1, 73):
#         next_aqi = model.predict(current_features.to_frame().T)[0]
#         predictions.append(next_aqi)

#         prediction_time = current_time + timedelta(hours=i)
#         timestamps.append(prediction_time.strftime('%Y-%m-%d %H:%M:%S'))

#         # Update cyclic hour + DOW
#         hour = prediction_time.hour
#         day_of_week = prediction_time.weekday()
#         current_features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
#         current_features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
#         for d in range(7):
#             current_features[f"dow_{d}"] = 1 if d == day_of_week else 0

#     forecast_df = pd.DataFrame({
#         "timestamp": timestamps,
#         "predicted_AQI": np.round(predictions, 2)
#     })

#     return {
#         "message": "✅ 72-hour AQI forecast generated successfully!",
#         "forecast": forecast_df.to_dict(orient="records")
#     }


# #----------------------------NEW CODE VERSION BELOW----------------------------#

# # ==============================================
# # 🌐 FastAPI Backend for AQI Prediction (Hopsworks + XGBoost)
# # ==============================================

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pandas as pd
# import numpy as np
# import hopsworks
# import joblib
# import os
# import math
# from datetime import datetime, timedelta
# from dotenv import load_dotenv

# # ==============================================
# # 1️⃣ Environment Setup
# # ==============================================
# load_dotenv()
# API_KEY = os.getenv("HOPSWORKS_API_KEY")
# PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT")

# # ==============================================
# # 2️⃣ Connect to Hopsworks and Load Latest Model
# # ==============================================
# try:
#     project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
#     mr = project.get_model_registry()

#     # Get latest version of the registered XGBoost model
#     model_entry = mr.get_model("xgb_aqi_model", version=None)  # version=None = latest
#     model_dir = model_entry.download()
#     bundle = joblib.load(model_dir + "/model_bundle.pkl")
#     model = bundle["model"]
#     features = bundle["features"]

#     print(f"✅ Loaded model 'xgb_aqi_model' (features: {len(features)}) from Hopsworks Model Registry!")

# except Exception as e:
#     print("❌ Failed to load model from Hopsworks:", e)
#     raise e

# # ==============================================
# # 3️⃣ Initialize FastAPI App
# # ==============================================
# app = FastAPI(
#     title="🌍 AQI Prediction API",
#     description="Predict Air Quality Index (AQI) using an XGBoost model stored in Hopsworks Model Registry",
#     version="2.0.0"
# )

# # ==============================================
# # 4️⃣ Request Schema
# # ==============================================
# class AQIInput(BaseModel):
#     relative_humidity_2m: float
#     pm10: float
#     pm2_5: float
#     ozone: float
#     nitrogen_dioxide: float
#     hour: int
#     day_of_week: int
#     season: str   # 'spring', 'summer', 'winter'

# # ==============================================
# # 5️⃣ Helper: Encode Input for Model
# # ==============================================
# def encode_input(data: AQIInput) -> pd.DataFrame:
#     """Convert user input into ML model feature vector"""
#     hour_sin = math.sin(2 * math.pi * data.hour / 24)
#     hour_cos = math.cos(2 * math.pi * data.hour / 24)

#     # One-hot encoding for season
#     season_spring = 1 if data.season == "spring" else 0
#     season_summer = 1 if data.season == "summer" else 0
#     season_winter = 1 if data.season == "winter" else 0

#     # One-hot encoding for day of week (0-6)
#     dow_encoded = [1 if data.day_of_week == i else 0 for i in range(7)]

#     input_vector = pd.DataFrame([[
#         data.relative_humidity_2m, data.pm10, data.pm2_5, data.ozone, data.nitrogen_dioxide,
#         season_spring, season_summer, season_winter,
#         hour_sin, hour_cos,
#         *dow_encoded
#     ]], columns=features)

#     return input_vector

# # ==============================================
# # 6️⃣ Root Endpoint
# # ==============================================
# @app.get("/")
# def root():
#     return {
#         "message": "🌍 AQI Prediction API is running!",
#         "usage": "POST /predict or /forecast_72hr with required features.",
#         "example_input": {
#             "relative_humidity_2m": 60,
#             "pm10": 40,
#             "pm2_5": 25,
#             "ozone": 35,
#             "nitrogen_dioxide": 18,
#             "hour": 14,
#             "day_of_week": 3,
#             "season": "summer"
#         }
#     }

# # ==============================================
# # 7️⃣ Predict AQI (Single)
# # ==============================================
# @app.post("/predict")
# def predict_aqi(data: AQIInput):
#     try:
#         input_vector = encode_input(data)
#         pred_aqi = float(model.predict(input_vector)[0])
#         return {"predicted_AQI": round(pred_aqi, 2)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# # ==============================================
# # 8️⃣ Forecast for Next 72 Hours
# # ==============================================
# @app.post("/forecast_72hr")
# def forecast_72hr(data: AQIInput):
#     try:
#         current_features = encode_input(data).iloc[0].copy()
#         current_time = datetime.now()
#         predictions, timestamps = [], []

#         for i in range(1, 73):
#             next_aqi = model.predict(current_features.to_frame().T)[0]
#             predictions.append(round(next_aqi, 2))
#             prediction_time = current_time + timedelta(hours=i)
#             timestamps.append(prediction_time.strftime("%Y-%m-%d %H:%M:%S"))

#             # Update time features
#             hour = prediction_time.hour
#             dow = prediction_time.weekday()
#             current_features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
#             current_features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
#             for d in range(7):
#                 current_features[f"dow_{d}"] = 1 if d == dow else 0

#         forecast_df = pd.DataFrame({"timestamp": timestamps, "predicted_AQI": predictions})
#         return {"message": "✅ 72-hour AQI forecast generated successfully!", "forecast": forecast_df.to_dict(orient="records")}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Forecast generation failed: {e}")


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
mr = project.get_model_registry()

# ==============================================================
# 3️⃣ Find the Best Model (Highest R²)
# ==============================================================
model_names = ["xgb_aqi_model", "ridge_aqi_model", "lstm_aqi_model"]

best_model_meta = None
best_model_name = None
best_model_type = None
best_r2 = -999

# for name in model_names:
#     try:
#         models = mr.get_models(name=name)
#         for m in models:
#             if m.metrics and "r2" in m.metrics:
#                 if m.metrics["r2"] > best_r2:
#                     best_r2 = m.metrics["r2"]
#                     best_model_meta = m
#                     best_model_name = name
#                     best_model_type = (
#                         "lstm" if "lstm" in name else
#                         "ridge" if "ridge" in name else
#                         "xgb"
#                     )
#     except Exception as e:
#         print(f"⚠️ Could not read model {name}: {e}")

# if best_model_meta is None:
#     raise ValueError("❌ No model with R² found in Hopsworks Model Registry!")

# another version
# for name in model_names:
#     try:
#         # Get the latest version of the model
#         m = mr.get_model(name, version=None)  # latest version
#         metrics = getattr(m, "metrics", None)
        
#         if metrics and "r2" in metrics:
#             r2 = metrics["r2"]
#             if r2 > best_r2:
#                 best_r2 = r2
#                 best_model_meta = m
#                 best_model_name = name
#                 best_model_type = (
#                     "lstm" if "lstm" in name else
#                     "ridge" if "ridge" in name else
#                     "xgb"
#                 )
#     except Exception as e:
#         print(f"⚠️ Could not read model {name}: {e}")

# if best_model_meta is None:
#     raise ValueError("❌ No model with R² found among latest versions of the 3 models!")

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
        
        # if metrics and "r2" in metrics:
        #     r2 = metrics["r2"]
        #     if r2 > best_r2:
        #         best_r2 = r2
        #         best_model_meta = model_version
        #         best_model_name = name
        #         best_model_type = (
        #             "lstm" if "lstm" in name else
        #             "ridge" if "ridge" in name else
        #             "xgb"
        #         )
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

# if best_model_type == "lstm":
#     model = load_model(os.path.join(model_dir, "lstm_model.h5"))
#     features = [
#         'relative_humidity_2m', 'pm10', 'pm2_5', 'ozone', 'nitrogen_dioxide',
#         'season_spring', 'season_summer', 'season_winter',
#         'hour_sin', 'hour_cos',
#         'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6'
#     ]
# import os
# if best_model_type == "lstm":
#     model = load_model(os.path.join(model_dir, "lstm_model.h5"))
#     features = [
#         'pm10', 'pm2_5', 'ozone', 'nitrogen_dioxide',
#         'season_spring', 'season_summer', 'season_winter',
#         'hour_sin', 'hour_cos',
#         'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6'
#     ]
# else:
#     # bundle = joblib.load(
#     #     os.path.join(model_dir, f"{best_model_type}_model.pkl")
#     # )
#     # model = bundle["model"]
#     # features = bundle["features"]
#     # Download model
#     model_dir = best_model_meta.download()  # returns local folder path
#     print("Files in model_dir:", os.listdir(model_dir))
#     best_model_file=os.path.join(model_dir, f"{best_model_type}_model.pkl")
#     bundle = joblib.load(best_model_file)
#     model = bundle["model"]
#     features = bundle["features"]



# print("✅ Model loaded successfully!")

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

# ==============================================================
# 6️⃣ Helper: Encode Input Features
# ==============================================================
def preprocess_input(request: AQIRequest):
    data = {
        # "relative_humidity_2m": request.relative_humidity_2m,
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

# ==============================================================
# 8️⃣ Predict Single AQI
# ==============================================================
@app.post("/predict")
def predict(request: AQIRequest):
    features_dict = preprocess_input(request)
    df = pd.DataFrame([features_dict])[features]

    if best_model_type == "lstm":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df)
        X_reshaped = np.expand_dims(X_scaled, axis=0)
        pred = model.predict(X_reshaped)[0][0]
    else:
        pred = model.predict(df)[0]

    return {
        "predicted_AQI": float(pred),
        "model_used": best_model_name,
        "r2": best_r2
    }

# ==============================================================
# 9️⃣ Predict Next 3 Days AQI (Non-Autoregressive)
# ==============================================================
@app.post("/forecast_3day")
def forecast_3day(request: AQIRequest):
    base_features = preprocess_input(request)
    now = datetime.utcnow()
    forecasts = []

    for i in range(1, 4):
        future = now + timedelta(days=i)
        fdict = base_features.copy()

        # Update temporal encodings for future day
        fdict["hour_sin"] = np.sin(2 * np.pi * future.hour / 24)
        fdict["hour_cos"] = np.cos(2 * np.pi * future.hour / 24)
        dow = future.weekday()
        for d in range(7):
            fdict[f"dow_{d}"] = 1 if d == dow else 0
        month = future.month
        fdict["season_spring"] = 1 if month in [3, 4, 5] else 0
        fdict["season_summer"] = 1 if month in [6, 7, 8] else 0
        fdict["season_winter"] = 1 if month in [12, 1, 2] else 0

        df_future = pd.DataFrame([fdict])[features]

        if best_model_type == "lstm":
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(df_future)
            X_reshaped = np.expand_dims(X_scaled, axis=0)
            pred = model.predict(X_reshaped)[0][0]
        else:
            pred = model.predict(df_future)[0]

        forecasts.append({
            "day": f"Day {i}",
            "date": future.strftime("%Y-%m-%d"),
            "predicted_AQI": float(pred)
        })

    # return {
    #     "forecast": forecasts,
    #     "model_used": best_model_name,
    #     "model_version": best_model_meta.version,
    #     "best_r2": best_r2
    # }
    return {
    "forecast": forecast_df.to_dict(orient="records"),
    "model_used": best_model.name if best_model else "unknown",
    "version": best_model.version if best_model else "N/A",
    "r2": best_r2 if best_r2 is not None else "N/A"
}


print("🚀 API ready — serving best model automatically!")
