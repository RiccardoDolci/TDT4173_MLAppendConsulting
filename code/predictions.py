# Import required libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

print("--- Starting Prediction Script ---")

# --- 1. Load Model and Data ---
print("Loading trained model and historical data...")

# This authenticates you automatically inside an Azure ML environment
ml_client = MLClient.from_config(credential=DefaultAzureCredential())
# Define paths
outputs_dir = Path('outputs')
model_dir = outputs_dir / 'model'
data_dir = Path('data')

# Load the trained model
try:
    model = joblib.load(model_dir / 'best_lgbm_model.joblib')
except FileNotFoundError:
    print(f"ERROR: Model file not found at '{model_dir / 'best_lgbm_model.joblib'}'. Please run the training script first.")
    exit()
# --- Load the dataset from the Azure ML Data Asset ---
# The name and specific version number from the URI
asset_name = "model_ready_data"
asset_version = "3"

# Get the data asset using its exact name and version
data_asset = ml_client.data.get(name=asset_name, version=asset_version)

# Read the data into pandas from the cloud path
historical_df = pd.read_csv(data_asset.path)
print("Successfully loaded data from Azure.")

# Load the prediction mapping file which defines the required output
try:
    prediction_mapping = pd.read_csv('prediction_mapping.csv')
except FileNotFoundError:
    print(f"ERROR: 'sample_submission.csv' not found")
    exit()


print("Model and data loaded successfully.")

# --- 2. Prepare Data and Holiday Calendar ---
historical_df['date_arrival'] = pd.to_datetime(historical_df['date_arrival'])
all_rm_ids = historical_df['rm_id'].unique()

# Get the exact feature list from the trained model to ensure consistency
features_to_use = model.feature_name_
print(f"Model expects the following features: {features_to_use}")

# *** THE FIX IS HERE: Define the list of categorical columns ***
categorical_cols = ['rm_id', 'month', 'day', 'day_of_week', 'week_of_year', 'is_closure_day']

live_data_df = historical_df.copy()

# ** Learn recurring holiday/closure patterns from historical data **
print("Learning historical closure patterns...")
daily_total_weight = historical_df.groupby('date_arrival')['net_weight'].sum()
closure_dates = daily_total_weight[daily_total_weight == 0].index
# Only learn holidays from non-weekends to avoid capturing every Saturday/Sunday
is_weekend_mask = closure_dates.dayofweek.isin([5, 6])
actual_holidays = closure_dates[~is_weekend_mask]
recurring_holidays = set(zip(actual_holidays.month, actual_holidays.day))
print(f"Identified {len(recurring_holidays)} unique recurring non-weekend holiday/closure days.")


# --- 3. REWRITTEN Recursive Forecasting Loop (More Robust and Memory-Efficient) ---
print("\n--- Starting Recursive Forecasting for 2025 ---")
forecast_dates = pd.date_range(start='2025-01-01', end='2025-05-31', freq='D')
future_features_log = []

for current_date in forecast_dates:
    print(f"Predicting for: {current_date.strftime('%Y-%m-%d')}")
    
    features_to_predict = pd.DataFrame({'date_arrival': [current_date] * len(all_rm_ids), 'rm_id': all_rm_ids})

    # Create static date features
    features_to_predict['month'] = features_to_predict['date_arrival'].dt.month
    features_to_predict['day'] = features_to_predict['date_arrival'].dt.day
    features_to_predict['day_of_week'] = features_to_predict['date_arrival'].dt.dayofweek
    features_to_predict['week_of_year'] = features_to_predict['date_arrival'].dt.isocalendar().week
    is_weekend = features_to_predict['day_of_week'].isin([5, 6])
    is_holiday = features_to_predict.apply(lambda row: (row['month'], row['day']) in recurring_holidays, axis=1)
    features_to_predict['is_closure_day'] = (is_weekend | is_holiday).astype(int)

    # Use efficient .map() with proper aggregation to create dynamic features
    last_delivery_dates_map = live_data_df[live_data_df['net_weight'] > 0].groupby('rm_id')['date_arrival'].max()
    
    # Get last known state for cumulative features
    last_known_state = live_data_df.groupby('rm_id').last()
    last_yearly_cumsum_map = last_known_state['yearly_cum_net_weight']
    last_monthly_cumsum_map = last_known_state['monthly_cum_net_weight']

    features_to_predict['last_delivery_date'] = features_to_predict['rm_id'].map(last_delivery_dates_map)
    features_to_predict['time_since_last_delivery'] = (features_to_predict['date_arrival'] - features_to_predict['last_delivery_date']).dt.days
    
    # Set initial cumulative values from last known state
    features_to_predict['yearly_cum_net_weight'] = features_to_predict['rm_id'].map(last_yearly_cumsum_map)
    features_to_predict['monthly_cum_net_weight'] = features_to_predict['rm_id'].map(last_monthly_cumsum_map)
    
    # Handle the reset logic for the new year/month
    if current_date.day == 1:
        features_to_predict['monthly_cum_net_weight'] = 0
    if current_date.day == 1 and current_date.month == 1:
        features_to_predict['yearly_cum_net_weight'] = 0

    for lag in [7, 14, 28]:
        lag_date = current_date - pd.Timedelta(days=lag)
        lag_map = live_data_df[live_data_df['date_arrival'] == lag_date].groupby('rm_id')['net_weight'].sum()
        features_to_predict[f'lag_{lag}_days'] = features_to_predict['rm_id'].map(lag_map)

    for window in [7, 14, 28]:
        start_date = current_date - pd.Timedelta(days=window)
        window_data = live_data_df[(live_data_df['date_arrival'] >= start_date) & (live_data_df['date_arrival'] < current_date)]
        rolling_stats = window_data.groupby('rm_id')['net_weight'].agg(['mean', 'std']).rename(columns={'mean': f'rolling_mean_{window}_days', 'std': f'rolling_std_{window}_days'})
        features_to_predict = features_to_predict.merge(rolling_stats, on='rm_id', how='left')
        
    features_to_predict.drop(columns=['last_delivery_date'], inplace=True)
    
    time_since_na = features_to_predict['time_since_last_delivery'].isna()
    features_to_predict.fillna(0, inplace=True)
    features_to_predict.loc[time_since_na, 'time_since_last_delivery'] = 999
    features_to_predict['time_since_last_delivery'] = features_to_predict['time_since_last_delivery'].clip(upper=999)

    # Make Prediction
    for col in categorical_cols:
        features_to_predict[col] = features_to_predict[col].astype('category')
    
    X_today = features_to_predict[features_to_use]
    future_features_log.append(X_today)
    
    daily_preds = model.predict(X_today)
    daily_preds[daily_preds < 0] = 0
    
    # Update History
    features_to_predict['net_weight'] = daily_preds
    features_to_predict['yearly_cum_net_weight'] = features_to_predict['yearly_cum_net_weight'] + daily_preds
    features_to_predict['monthly_cum_net_weight'] = features_to_predict['monthly_cum_net_weight'] + daily_preds
    features_to_predict['year'] = current_date.year # Add year back in for concat
    
    live_data_df = pd.concat([live_data_df, features_to_predict[historical_df.columns]], ignore_index=True)

# --- 4. Format Final Submission ---
print("\n--- Formatting Final Submission File ---")
predictions_2025 = live_data_df[live_data_df['date_arrival'] >= '2025-01-01']
# The required prediction is the overall cumulative sum for the forecast period
predictions_2025['prediction'] = predictions_2025.groupby('rm_id')['net_weight'].cumsum()

prediction_mapping_renamed = prediction_mapping.rename(columns={'ID': 'id', 'forecast_end_date': 'date_arrival'})
prediction_mapping_renamed['date_arrival'] = pd.to_datetime(prediction_mapping_renamed['date_arrival'])
submission = pd.merge(prediction_mapping_renamed, predictions_2025[['date_arrival', 'rm_id', 'prediction']], on=['date_arrival', 'rm_id'], how='left')
submission = submission[['id', 'prediction']].fillna(0)
submission_path = outputs_dir / 'submission.csv'
submission.to_csv(submission_path, index=False)
print(f"Submission file created successfully and saved to: {submission_path}")

# --- 5. Save the Future Features for Debugging ---
print("\n--- Saving Future Prediction Features for Debugging ---")
all_future_features_df = pd.concat(future_features_log, ignore_index=True)
future_features_path = outputs_dir / 'future_predictions_features.csv'
all_future_features_df.to_csv(future_features_path, index=False)
print(f"Features used for 2025 predictions saved to: {future_features_path}")
