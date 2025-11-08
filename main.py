# import hopsworks

# project = hopsworks.login(api_key_value="", project="")  # interactive or env
# # fs = project.get_feature_store()
# # print("Connected to feature store:", fs)
# fs = project.get_feature_store()

# raw_fg = fs.get_or_create_feature_group(
#     name="aqi_raw_weather_pollution",
#     version=1,
#     primary_key=["timestamp"],
#     description="Raw hourly weather and pollution data fetched from OpenMeteo APIs"
# )

# raw_fg.insert(merged_df, write_options={"wait_for_job": False})

# processed_fg = fs.get_or_create_feature_group(
#     name="aqi_hourly_features",
#     version=1,
#     primary_key=["timestamp"],
#     description="Processed and engineered hourly features with AQI target"
# )

# processed_fg.insert(final_df, write_options={"wait_for_job": False})


#   BELOW SCRIPT IS FOR INSPECTING THE FEATURE GROUPS


import hopsworks
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

import hopsworks, pandas as pd
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")

project = hopsworks.login(api_key_value=api_key, project=project_name)
fs = project.get_feature_store()

def inspect_fg(name, version=1):
    fg = fs.get_feature_group(name, version)
    df = fg.read()
    print(f"\n=== {name.upper()} ===")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Date range:", df['timestamp'].min(), "→", df['timestamp'].max())
    print("Duration:", (pd.to_datetime(df['timestamp']).max() - pd.to_datetime(df['timestamp']).min()))
    print("Missing values:\n", df.isna().sum())
    print("----")

inspect_fg("aqi_raw_weather_pollution")
inspect_fg("aqi_hourly_features", version=2)

# print("Total rows:", len(raw_df))
print("Unique timestamps:", raw_df['timestamp'].nunique())

# Below will print structure of project
# import os

# def print_folder_structure(root_dir):
#     for root, dirs, files in os.walk(root_dir):
#         level = root.replace(root_dir, '').count(os.sep)
#         indent = ' ' * 4 * level
#         print(f"{indent}{os.path.basename(root)}/")
#         sub_indent = ' ' * 4 * (level + 1)
#         for f in files:
#             print(f"{sub_indent}{f}")

# # Example usage:
# print_folder_structure("D:/AQI_Predictor")
