# ==============================================
# 🚀 AQI Model Training Pipeline with Hopsworks
# ==============================================

import hopsworks
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import hopsworks
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")


# ==============================================
# 1️⃣ Connect to Hopsworks and Load Data
# ==============================================
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"), project="AQI_Predictor10pearls")
fs = project.get_feature_store()

#version 2 use kia hai
fg = fs.get_feature_group("aqi_hourly_features", version=2)
df = fg.read()

print("✅ Data loaded from Hopsworks Feature Store!")
print("Shape:", df.shape)
# print("Columns in Feature Group:")
# print(df.columns)


# Sort by timestamp to maintain chronological order
df = df.sort_values("timestamp").reset_index(drop=True)

# ==============================================
# 2️⃣ Feature / Target Separation
# ==============================================
features = [
    'relative_humidity_2m', 'pm10', 'pm2_5', 'ozone', 'nitrogen_dioxide',
    'season_spring', 'season_summer', 'season_winter',
    'hour_sin', 'hour_cos',
    'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6'
]
target = 'aqi'

X = df[features]
y = df[target]

# ==============================================
# 3️⃣ Time-Series Cross-Validation
# ==============================================
print("\n⏳ Performing Time-Series Cross Validation...")

tscv = TimeSeriesSplit(n_splits=5)
fold_results = []
fold_idx = 1

for train_index, val_index in tscv.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    y_pred = model.predict(X_val)
    fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    fold_mae = mean_absolute_error(y_val, y_pred)
    fold_r2 = r2_score(y_val, y_pred)

    fold_results.append((fold_rmse, fold_mae, fold_r2))
    print(f"Fold {fold_idx} → RMSE: {fold_rmse:.2f}, MAE: {fold_mae:.2f}, R²: {fold_r2:.3f}")
    fold_idx += 1

# Average CV results
avg_rmse = np.mean([r[0] for r in fold_results])
avg_mae = np.mean([r[1] for r in fold_results])
avg_r2 = np.mean([r[2] for r in fold_results])

print("\n📈 Cross-Validation Summary:")
print(f"Avg RMSE: {avg_rmse:.2f}")
print(f"Avg MAE:  {avg_mae:.2f}")
print(f"Avg R²:   {avg_r2:.3f}")

# ==============================================
# 4️⃣ Final Train/Test Split
# ==============================================
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# ==============================================
# 5️⃣ Train Final XGBoost with Early Stopping
# ==============================================
final_model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="rmse"
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=False
)

print("✅ Final model trained successfully!")

# ==============================================
# 6️⃣ Evaluate Model
# ==============================================
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)


train_r2 = r2_score(y_train, y_train_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\n🎯 Final Model Evaluation:")
print(f"Train R²:   {train_r2:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test MAE:  {mae:.3f}")
print(f"Test R²:   {r2:.3f}")

# ==============================================
# 7️⃣ Save Model Bundle Locally
# ==============================================
bundle = {
    "model": final_model,
    "features": features,
    "metrics": {"cv_rmse": avg_rmse, "cv_mae": avg_mae, "cv_r2": avg_r2,
                "test_rmse": rmse, "test_mae": mae, "test_r2": r2}
}

os.makedirs("artifacts", exist_ok=True)
joblib.dump(bundle, "artifacts/model_bundle.pkl")
print("💾 Model bundle saved to artifacts/model_bundle.pkl")

# # ==============================================
# # 8️⃣ Log Model to Hopsworks Model Registry
# # ==============================================
# mr = project.get_model_registry()

# model_metadata = mr.log_model(
#     name="xgb_aqi_model",
#     metrics={
#         "cv_rmse": float(avg_rmse),
#         "cv_mae": float(avg_mae),
#         "cv_r2": float(avg_r2),
#         "test_rmse": float(rmse),
#         "test_mae": float(mae),
#         "test_r2": float(r2)
#     },
#     model_file="artifacts/model_bundle.pkl",
#     description="XGBoost AQI model with time-series CV and early stopping"
# )

# print(f"✅ Model logged to Hopsworks Registry! Version: {model_metadata.version}")


# ==============================================
# 8️⃣ Log Model to Hopsworks Model Registry (final fixed version)
# ==============================================
from hsml import schema
from hsml.model_schema import ModelSchema  # ✅ Correct import

mr = project.get_model_registry()

# Define input/output schemas
input_schema = schema.Schema(X_train)
output_schema = schema.Schema(y_train)

# Create model schema
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

# Register the model in Hopsworks
model_metadata = mr.python.create_model(
    name="xgb_aqi_model",
    description="XGBoost AQI model with time-series CV and early stopping",
    metrics={
        "cv_rmse": float(avg_rmse),
        "cv_mae": float(avg_mae),
        "cv_r2": float(avg_r2),
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "test_r2": float(r2)
    },
    model_schema=model_schema,
)

# Save the model file to the registry
model_metadata.save("artifacts/model_bundle.pkl")
# model_metadata.save()----------redundant line hata di

print(f"✅ Model logged to Hopsworks Registry! Version: {model_metadata.version}")
