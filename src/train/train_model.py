# # ==============================================
# # 🚀 AQI Model Training Pipeline with Hopsworks
# # ==============================================

# import hopsworks
# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# import os
# import hopsworks
# from dotenv import load_dotenv
# import os
# load_dotenv()
# api_key = os.getenv("HOPSWORKS_API_KEY")
# project_name = os.getenv("HOPSWORKS_PROJECT")


# # ==============================================
# # 1️⃣ Connect to Hopsworks and Load Data
# # ==============================================
# project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"), project="AQI_Predictor10pearls")
# fs = project.get_feature_store()

# #version 2 use kia hai
# fg = fs.get_feature_group("aqi_hourly_features", version=2)
# df = fg.read()

# print("✅ Data loaded from Hopsworks Feature Store!")
# print("Shape:", df.shape)
# # print("Columns in Feature Group:")
# # print(df.columns)


# # Sort by timestamp to maintain chronological order
# df = df.sort_values("timestamp").reset_index(drop=True)

# # ==============================================
# # 2️⃣ Feature / Target Separation
# # ==============================================
# features = [
#     'relative_humidity_2m', 'pm10', 'pm2_5', 'ozone', 'nitrogen_dioxide',
#     'season_spring', 'season_summer', 'season_winter',
#     'hour_sin', 'hour_cos',
#     'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6'
# ]
# target = 'aqi'

# X = df[features]
# y = df[target]

# # ==============================================
# # 3️⃣ Time-Series Cross-Validation
# # ==============================================
# print("\n⏳ Performing Time-Series Cross Validation...")

# tscv = TimeSeriesSplit(n_splits=5)
# fold_results = []
# fold_idx = 1

# for train_index, val_index in tscv.split(X):
#     X_train, X_val = X.iloc[train_index], X.iloc[val_index]
#     y_train, y_val = y.iloc[train_index], y.iloc[val_index]

#     model = XGBRegressor(
#         n_estimators=2000,
#         learning_rate=0.01,
#         max_depth=3,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         min_child_weight=5,
#         gamma=0.2,
#         reg_alpha=1.0,
#         reg_lambda=2.0,
#         random_state=42,
#         n_jobs=-1,
#         eval_metric="rmse"
#     )

#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         early_stopping_rounds=50,
#         verbose=False
#     )

#     y_pred = model.predict(X_val)
#     fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
#     fold_mae = mean_absolute_error(y_val, y_pred)
#     fold_r2 = r2_score(y_val, y_pred)

#     fold_results.append((fold_rmse, fold_mae, fold_r2))
#     print(f"Fold {fold_idx} → RMSE: {fold_rmse:.2f}, MAE: {fold_mae:.2f}, R²: {fold_r2:.3f}")
#     fold_idx += 1

# # Average CV results
# avg_rmse = np.mean([r[0] for r in fold_results])
# avg_mae = np.mean([r[1] for r in fold_results])
# avg_r2 = np.mean([r[2] for r in fold_results])

# print("\n📈 Cross-Validation Summary:")
# print(f"Avg RMSE: {avg_rmse:.2f}")
# print(f"Avg MAE:  {avg_mae:.2f}")
# print(f"Avg R²:   {avg_r2:.3f}")

# # ==============================================
# # 4️⃣ Final Train/Test Split
# # ==============================================
# split_index = int(len(df) * 0.8)
# X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
# y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# # ==============================================
# # 5️⃣ Train Final XGBoost with Early Stopping
# # ==============================================
# final_model = XGBRegressor(
#     n_estimators=2000,
#     learning_rate=0.01,
#     max_depth=3,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     min_child_weight=5,
#     gamma=0.2,
#     reg_alpha=1.0,
#     reg_lambda=2.0,
#     random_state=42,
#     n_jobs=-1,
#     eval_metric="rmse"
# )

# final_model.fit(
#     X_train, y_train,
#     eval_set=[(X_test, y_test)],
#     early_stopping_rounds=50,
#     verbose=False
# )

# print("✅ Final model trained successfully!")

# # ==============================================
# # 6️⃣ Evaluate Model
# # ==============================================
# y_train_pred = final_model.predict(X_train)
# y_test_pred = final_model.predict(X_test)


# train_r2 = r2_score(y_train, y_train_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
# mae = mean_absolute_error(y_test, y_test_pred)
# r2 = r2_score(y_test, y_test_pred)

# print("\n🎯 Final Model Evaluation:")
# print(f"Train R²:   {train_r2:.3f}")
# print(f"Test RMSE: {rmse:.3f}")
# print(f"Test MAE:  {mae:.3f}")
# print(f"Test R²:   {r2:.3f}")

# # ==============================================
# # 7️⃣ Save Model Bundle Locally
# # ==============================================
# bundle = {
#     "model": final_model,
#     "features": features,
#     "metrics": {"cv_rmse": avg_rmse, "cv_mae": avg_mae, "cv_r2": avg_r2,
#                 "test_rmse": rmse, "test_mae": mae, "test_r2": r2}
# }

# os.makedirs("artifacts", exist_ok=True)
# joblib.dump(bundle, "artifacts/model_bundle.pkl")
# print("💾 Model bundle saved to artifacts/model_bundle.pkl")

# # # ==============================================
# # # 8️⃣ Log Model to Hopsworks Model Registry
# # # ==============================================
# # mr = project.get_model_registry()

# # model_metadata = mr.log_model(
# #     name="xgb_aqi_model",
# #     metrics={
# #         "cv_rmse": float(avg_rmse),
# #         "cv_mae": float(avg_mae),
# #         "cv_r2": float(avg_r2),
# #         "test_rmse": float(rmse),
# #         "test_mae": float(mae),
# #         "test_r2": float(r2)
# #     },
# #     model_file="artifacts/model_bundle.pkl",
# #     description="XGBoost AQI model with time-series CV and early stopping"
# # )

# # print(f"✅ Model logged to Hopsworks Registry! Version: {model_metadata.version}")


# # ==============================================
# # 8️⃣ Log Model to Hopsworks Model Registry (final fixed version)
# # ==============================================
# from hsml import schema
# from hsml.model_schema import ModelSchema  # ✅ Correct import

# mr = project.get_model_registry()

# # Define input/output schemas
# input_schema = schema.Schema(X_train)
# output_schema = schema.Schema(y_train)

# # Create model schema
# model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

# # Register the model in Hopsworks
# model_metadata = mr.python.create_model(
#     name="xgb_aqi_model",
#     description="XGBoost AQI model with time-series CV and early stopping",
#     metrics={
#         "cv_rmse": float(avg_rmse),
#         "cv_mae": float(avg_mae),
#         "cv_r2": float(avg_r2),
#         "test_rmse": float(rmse),
#         "test_mae": float(mae),
#         "test_r2": float(r2)
#     },
#     model_schema=model_schema,
# )

# # Save the model file to the registry
# model_metadata.save("artifacts/model_bundle.pkl")
# # model_metadata.save()----------redundant line hata di

# print(f"✅ Model logged to Hopsworks Registry! Version: {model_metadata.version}")

# ==============================================
# 🚀 AQI Model Training Pipeline with Hopsworks
# Supports: XGBoost, Ridge Regression, LSTM
# ==============================================

import hopsworks
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, os
from dotenv import load_dotenv
from hsml import schema
from hsml.model_schema import ModelSchema
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# ==============================================
# 1️⃣ Connect to Hopsworks and Load Data
# ==============================================
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")

project = hopsworks.login(api_key_value=api_key, project=project_name)
fs = project.get_feature_store()
mr = project.get_model_registry()

fg = fs.get_feature_group("aqi_hourly_features", version=2)
df = fg.read()

print("✅ Data loaded from Hopsworks Feature Store!")
print("Shape:", df.shape)

# ==============================================
# 2️⃣ Preprocess for Model Training
# ==============================================
df = df.sort_values("timestamp").reset_index(drop=True)

# features = [
#     'relative_humidity_2m', 'pm10', 'pm2_5', 'ozone', 'nitrogen_dioxide',
#     'season_spring', 'season_summer', 'season_winter',
#     'hour_sin', 'hour_cos',
#     'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6'
# ]
# without relative humidity
features = [
    'pm10', 'pm2_5', 'ozone', 'nitrogen_dioxide',
    'season_spring', 'season_summer', 'season_winter',
    'hour_sin', 'hour_cos',
    'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6'
]
target = 'aqi'

X = df[features]
y = df[target]

# ==============================================
# 3️⃣ Time-Series Cross Validation (Shared)
# ==============================================
print("\n⏳ Performing Time-Series Cross Validation...")

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = {"xgb": [], "ridge": []}

# --- XGBoost Cross Validation ---
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    xgb = XGBRegressor(
        n_estimators=1500, learning_rate=0.01, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, eval_metric="rmse"
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)],
            early_stopping_rounds=50, verbose=False)
    y_pred = xgb.predict(X_val)
    cv_scores["xgb"].append(np.sqrt(mean_squared_error(y_val, y_pred)))

# --- Ridge Regression Cross Validation ---
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_val)
    cv_scores["ridge"].append(np.sqrt(mean_squared_error(y_val, y_pred)))

print(f"📈 XGBoost CV RMSE: {np.mean(cv_scores['xgb']):.3f}")
print(f"📉 Ridge CV RMSE: {np.mean(cv_scores['ridge']):.3f}")

# ==============================================
# 4️⃣ Train/Test Split (Final)
# ==============================================
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# ==============================================
# 5️⃣ Train Final XGBoost Model
# ==============================================
xgb_model = XGBRegressor(
    n_estimators=2000, learning_rate=0.01, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    n_jobs=-1, eval_metric="rmse"
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              early_stopping_rounds=50, verbose=False)

y_pred_xgb = xgb_model.predict(X_test)
metrics_xgb = {
    "rmse": np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    "mae": mean_absolute_error(y_test, y_pred_xgb),
    "r2": r2_score(y_test, y_pred_xgb)
}

print("\n✅ XGBoost Model Trained:")
print(metrics_xgb)

# ==============================================
# 6️⃣ Train Ridge Regression
# ==============================================
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
metrics_ridge = {
    "rmse": np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    "mae": mean_absolute_error(y_test, y_pred_ridge),
    "r2": r2_score(y_test, y_pred_ridge)
}

print("\n✅ Ridge Regression Model Trained:")
print(metrics_ridge)

# ==============================================
# 7️⃣ Train LSTM Model
# ==============================================
print("\n🧠 Training LSTM Model (for sequential learning)...")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

# Create sequences (e.g., 24 hours = 1 day window)
window_size = 24
X_seq, y_seq = [], []
for i in range(len(X_scaled) - window_size):
    X_seq.append(X_scaled[i:i + window_size])
    y_seq.append(y_scaled[i + window_size])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

split = int(0.8 * len(X_seq))
X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

lstm_model = Sequential([
    LSTM(64, activation='tanh', input_shape=(window_size, X_seq.shape[2])),
    Dense(32, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=30, batch_size=32, verbose=0,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

y_pred_lstm = lstm_model.predict(X_test_seq)
metrics_lstm = {
    "rmse": np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm)),
    "mae": mean_absolute_error(y_test_seq, y_pred_lstm),
    "r2": r2_score(y_test_seq, y_pred_lstm)
}

print("\n✅ LSTM Model Trained:")
print(metrics_lstm)

# ==============================================
# 8️⃣ Save Model Bundles
# ==============================================
os.makedirs("artifacts", exist_ok=True)
joblib.dump({"model": xgb_model, "features": features}, "artifacts/xgb_model.pkl")
joblib.dump({"model": ridge_model, "features": features}, "artifacts/ridge_model.pkl")
lstm_model.save("artifacts/lstm_model.h5")

print("💾 All models saved in /artifacts folder.")

# ==============================================
# 9️⃣ Log to Hopsworks Model Registry
# ==============================================
input_schema = schema.Schema(X_train)
output_schema = schema.Schema(y_train)
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

# # --- Log XGBoost ---
# xgb_meta = mr.python.create_model(
#     name="xgb_aqi_model",
#     description="XGBoost AQI model with time-series CV and early stopping",
#     metrics=metrics_xgb,
#     model_schema=model_schema,
# )
# xgb_meta.save("artifacts/xgb_model.pkl")
# print(f"✅ XGBoost logged to Hopsworks (v{xgb_meta.version})")

# --- XGBoost ---
xgb_meta = mr.python.create_model(
    name="xgb_aqi_model",
    description="XGBoost AQI model with time-series CV and early stopping",
    metrics={
        "rmse": float(metrics_xgb["rmse"]),
        "mae": float(metrics_xgb["mae"]),
        "r2": float(metrics_xgb["r2"])
    },
    model_schema=model_schema
)
xgb_meta.save("artifacts/xgb_model.pkl")
print(f"✅ XGBoost logged to Hopsworks (v{xgb_meta.version})")

# # --- Log Ridge Regression ---
# ridge_meta = mr.python.create_model(
#     name="ridge_aqi_model",
#     description="Ridge Regression baseline model for AQI prediction",
#     metrics=metrics_ridge,
#     model_schema=model_schema,
# )
# ridge_meta.save("artifacts/ridge_model.pkl")
# print(f"✅ Ridge Regression logged to Hopsworks (v{ridge_meta.version})")

# --- Ridge Regression ---
ridge_meta = mr.python.create_model(
    name="ridge_aqi_model",
    description="Ridge Regression baseline model for AQI prediction",
    metrics={
        "rmse": float(metrics_ridge["rmse"]),
        "mae": float(metrics_ridge["mae"]),
        "r2": float(metrics_ridge["r2"])
    },
    model_schema=model_schema
)
ridge_meta.save("artifacts/ridge_model.pkl")
print(f"✅ Ridge Regression logged to Hopsworks (v{ridge_meta.version})")

# # --- Log LSTM ---
# lstm_meta = mr.python.create_model(
#     name="lstm_aqi_model",
#     description="LSTM deep learning model for sequential AQI forecasting",
#     metrics=metrics_lstm,
#     model_schema=model_schema,
# )
# lstm_meta.save("artifacts/lstm_model.h5")
# print(f"✅ LSTM Model logged to Hopsworks (v{lstm_meta.version})")

# --- LSTM ---
lstm_meta = mr.python.create_model(
    name="lstm_aqi_model",
    description="LSTM deep learning model for sequential AQI forecasting",
    metrics={
        "rmse": float(metrics_lstm["rmse"]),
        "mae": float(metrics_lstm["mae"]),
        "r2": float(metrics_lstm["r2"])
    },
    model_schema=model_schema
)
# Save LSTM in Keras recommended format
lstm_meta.save("artifacts/lstm_model.h5")
print(f"✅ LSTM logged to Hopsworks (v{lstm_meta.version})")
# ==============================================
# 📊 Final Summary of All Models
# ==============================================
print("\n==================== 🧠 FINAL MODEL PERFORMANCE SUMMARY ====================")
print(f"XGBoost  → RMSE: {metrics_xgb['rmse']:.3f}, MAE: {metrics_xgb['mae']:.3f}, R²: {metrics_xgb['r2']:.3f}")
print(f"Ridge    → RMSE: {metrics_ridge['rmse']:.3f}, MAE: {metrics_ridge['mae']:.3f}, R²: {metrics_ridge['r2']:.3f}")
print(f"LSTM     → RMSE: {metrics_lstm['rmse']:.3f}, MAE: {metrics_lstm['mae']:.3f}, R²: {metrics_lstm['r2']:.3f}")
try:
    print(f"RandomForest → RMSE: {rf_rmse:.3f}, MAE: {rf_mae:.3f}, R²: {rf_r2:.3f}")
except NameError:
    print("RandomForest → Not Trained in this run")
print("===========================================================================")


print("\n🎉 Training pipeline completed successfully!")

print("XGB metrics stored:", metrics_xgb)
print("Ridge metrics stored:", metrics_ridge)
print("LSTM metrics stored:", metrics_lstm)



for name in ["xgb_aqi_model", "ridge_aqi_model", "lstm_aqi_model"]:
    model_entry = mr.get_model(name)             # gets latest version entry
    version_number = model_entry.version
    version_obj = mr.get_model(name, version=version_number)

    metrics = getattr(version_obj, "training_metrics", None)
    print(f"{name} latest version: {version_number}, metrics: {metrics}")
