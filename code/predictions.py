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
asset_version = "8"

# Get the data asset using its exact name and version
data_asset = ml_client.data.get(name=asset_name, version=asset_version)

# Read the data into pandas from the cloud path
historical_df = pd.read_csv(data_asset.path, sep=',', thousands='.')
print("Successfully loaded data from Azure.")

# Lazy load fallback merged_clean_data (used to initialize features when historical_df lacks pre-start history)
merged_fallback = None
merged_path = Path('data') / 'mod_data' / 'merged_clean_data.csv'
if merged_path.exists():
    try:
        merged_fallback = pd.read_csv(merged_path)
        # Parse as UTC-aware timestamps and normalize to midnight UTC
        merged_fallback['date_arrival'] = pd.to_datetime(merged_fallback['date_arrival'], errors='coerce', utc=True)
        merged_fallback['date_arrival'] = merged_fallback['date_arrival'].dt.normalize()
        # aggregate net_weight per rm_id per date
        merged_daily = (
            merged_fallback.groupby(['rm_id', 'date_arrival'], as_index=False)['net_weight'].sum()
        )
        # compute yearly cumulative per rm_id
        merged_daily['year'] = merged_daily['date_arrival'].dt.year
        merged_daily = merged_daily.sort_values(['rm_id', 'date_arrival'])
        merged_daily['cum_net_weight'] = merged_daily.groupby(['rm_id', 'year'])['net_weight'].cumsum()
        # ensure merged_daily date dtype is timezone-aware UTC
        merged_daily['date_arrival'] = pd.to_datetime(merged_daily['date_arrival'], utc=True)
        merged_daily['date_arrival'] = merged_daily['date_arrival'].dt.normalize()
    except Exception:
        merged_fallback = None
        merged_daily = None
else:
    merged_daily = None

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

# *** Define the list of categorical columns ***
categorical_cols = ['rm_id', 'month', 'day', 'day_of_week', 'week_of_year', 'is_closure_day', 'first_day_of_year']

live_data_df = historical_df.copy()

# ** Learn recurring holiday/closure patterns from historical data **
print("Learning historical closure patterns...")
# Use the pre-computed 'is_closure_day' column from the model-ready data
# but only take values from 2024 (2024-01-01 .. 2024-12-31) as requested.
if 'is_closure_day' in historical_df.columns:
    closure_2024_df = (
        historical_df.loc[historical_df['date_arrival'].dt.year == 2024, ['date_arrival', 'is_closure_day']]
        .drop_duplicates(subset=['date_arrival'])
    )
    # map date_arrival (normalized) -> 0/1
    closure_2024_map = {
        pd.to_datetime(d).normalize().date(): int(v)
        for d, v in zip(closure_2024_df['date_arrival'], closure_2024_df['is_closure_day'])
    }
    num_closures = sum(1 for v in closure_2024_map.values() if v)
    print(f"Loaded is_closure_day for {len(closure_2024_map)} dates in 2024 ({num_closures} closure days).")
else:
    print("Warning: 'is_closure_day' not found in historical data; falling back to no closures.")
    closure_2024_map = {}


# --- 3. Dynamic multi-day forecasting per rm_id ---
print('\n--- Running dynamic forecasts according to prediction_mapping ---')
# Ensure dates in mapping are datetimes
# Make the mapping dates UTC-aware and normalized to midnight UTC for safe comparisons
prediction_mapping['forecast_start_date'] = pd.to_datetime(prediction_mapping['forecast_start_date'], utc=True).dt.normalize()
prediction_mapping['forecast_end_date'] = pd.to_datetime(prediction_mapping['forecast_end_date'], utc=True).dt.normalize()

# Use rows strictly before each rm_id's forecast start_date to build initial history
# (no global anchor). If not enough recent history exists, fall back to the same
# calendar window one year before, then to the last known row or zero.

# Prepare a lookup of historical last-known features per rm_id
# historical_df should contain the model-ready features up to its last date_arrival
# Ensure historical_df date_arrival is UTC-aware and normalized to midnight UTC
historical_df['date_arrival'] = pd.to_datetime(historical_df['date_arrival'], utc=True)
historical_df['date_arrival'] = historical_df['date_arrival'].dt.normalize()
last_known = historical_df.sort_values('date_arrival').groupby('rm_id').tail(1).set_index('rm_id')

results = []
debug_rows = []

# For each rm_id in mapping, simulate forward day-by-day and collect cum_net_weight at requested end dates
for rm_id, group in prediction_mapping.groupby('rm_id'):
    # determine forecast window for this rm_id
    start_date = group['forecast_start_date'].min().normalize()
    end_date = group['forecast_end_date'].max().normalize()

    # If rm_id not in historical data, skip with zeros
    if rm_id not in last_known.index:
        print(f"Warning: rm_id {rm_id} not in historical data — producing zeros for its forecasts")
        for _, row in group.iterrows():
            results.append({'ID': row['ID'], 'predicted_weight': 0.0})
        continue

    # Build rm-specific history and initialize state strictly from data BEFORE start_date
    hist_df_rm = historical_df.loc[historical_df['rm_id'] == rm_id].sort_values('date_arrival')

    # rows strictly before forecast start_date
    hist_before_start = hist_df_rm.loc[hist_df_rm['date_arrival'] < start_date]

    # Prefer the last-known row before start_date to initialize state
    if len(hist_before_start) > 0:
        state = hist_before_start.iloc[-1].to_dict()
    else:
        # Fallback: try the same calendar window one year before the start_date
        prev_year_anchor = start_date - pd.Timedelta(days=365)
        window_start = prev_year_anchor - pd.Timedelta(days=27)
        window_end = prev_year_anchor
        prev_year_window = hist_df_rm.loc[(hist_df_rm['date_arrival'] >= window_start) & (hist_df_rm['date_arrival'] <= window_end)]
        if len(prev_year_window) > 0:
            state = prev_year_window.iloc[-1].to_dict()
        else:
            # As a last resort, try to initialize from merged_clean_data fallback
            if merged_daily is not None:
                merged_rm = merged_daily.loc[merged_daily['rm_id'] == rm_id].sort_values('date_arrival')
                merged_before = merged_rm.loc[merged_rm['date_arrival'] < start_date]
                if len(merged_before) > 0:
                    # use last row before start_date
                    state = merged_before.iloc[-1].to_dict()
                    # set hist_values to last up-to-28 cum_net_weight values
                    hist_values = merged_before['cum_net_weight'].astype(float).tolist()[-28:]
                else:
                    if rm_id in last_known.index:
                        state = last_known.loc[rm_id].to_dict()
                    else:
                        state = {'cum_net_weight': 0.0}
            else:
                if rm_id in last_known.index:
                    state = last_known.loc[rm_id].to_dict()
                else:
                    state = {'cum_net_weight': 0.0}

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    date_preds = {}
    prev_cum = float(state.get('cum_net_weight', 0.0))

    hist_values = hist_before_start['cum_net_weight'].astype(float).tolist() if len(hist_before_start) > 0 else []
    if len(hist_values) < 1 and 'prev_year_window' in locals() and len(prev_year_window) > 0:
        hist_values = prev_year_window['cum_net_weight'].astype(float).tolist()

    # Ensure at least one value (the state cum_net_weight) is present and trim to 28
    if len(hist_values) == 0:
        hist_values = [prev_cum]
    else:
        if float(hist_values[-1]) != prev_cum:
            hist_values.append(prev_cum)
    hist_values = hist_values[-28:]

    for current_date in dates:
        # build feature row based on state and current_date
        feat = {}
        # populate date_arrival features
        feat['rm_id'] = rm_id
        feat['date_arrival'] = current_date
        feat['year'] = current_date.year
        feat['month'] = current_date.month
        feat['day'] = current_date.day
        feat['day_of_week'] = current_date.dayofweek
        # week_of_year may be an Index object in pandas >=1.1
        try:
            feat['week_of_year'] = int(current_date.isocalendar().week)
        except Exception:
            feat['week_of_year'] = int(current_date.isocalendar()[1])
        # closure/holiday feature derived from 2024 closure map loaded from historical data
        feat['is_closure_day'] = int(closure_2024_map.get(current_date.normalize().date(), 0))
        # first day of year flag
        feat['first_day_of_year'] = int((current_date.month == 1) and (current_date.day == 1))

        # Compute lag and rolling features from history (hist_values)
        # lag_1_day is previous day's cumulative
        lag_1 = float(hist_values[-1]) if len(hist_values) >= 1 else prev_cum
        lag_7 = float(hist_values[-7]) if len(hist_values) >= 7 else float(hist_values[0])
        lag_14 = float(hist_values[-14]) if len(hist_values) >= 14 else float(hist_values[0])
        lag_28 = float(hist_values[-28]) if len(hist_values) >= 28 else float(hist_values[0])

        # rolling means/std over previous windows (use available history)
        def safe_agg(values, window, func):
            vals = values[-window:] if len(values) >= 1 else [prev_cum]
            if len(vals) == 0:
                vals = [prev_cum]
            try:
                return float(func(vals))
            except Exception:
                return float(prev_cum)

        rolling_mean_7 = safe_agg(hist_values, 7, lambda v: sum(v) / len(v))
        rolling_mean_14 = safe_agg(hist_values, 14, lambda v: sum(v) / len(v))
        rolling_mean_28 = safe_agg(hist_values, 28, lambda v: sum(v) / len(v))

        import math
        def safe_std(values):
            if len(values) <= 1:
                return 0.0
            mean = sum(values) / len(values)
            return float(math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)))

        rolling_std_7 = safe_std(hist_values[-7:])
        rolling_std_14 = safe_std(hist_values[-14:])
        rolling_std_28 = safe_std(hist_values[-28:])

        # populate feature values
        feat['lag_1_day'] = lag_1
        feat['lag_7_days'] = lag_7
        feat['lag_14_days'] = lag_14
        feat['lag_28_days'] = lag_28
        feat['rolling_mean_7_days'] = rolling_mean_7
        feat['rolling_mean_14_days'] = rolling_mean_14
        feat['rolling_mean_28_days'] = rolling_mean_28
        feat['rolling_std_7_days'] = rolling_std_7
        feat['rolling_std_14_days'] = rolling_std_14
        feat['rolling_std_28_days'] = rolling_std_28

        # Ensure we produce a DataFrame with exactly the model features (fill missing with 0)
        X = pd.DataFrame([feat])
        for f in features_to_use:
            if f not in X.columns:
                X[f] = 0

        # Align column order
        X = X[features_to_use]

        # Ensure categorical dtypes/categories match training data to avoid LightGBM errors
        for cat in categorical_cols:
            if cat in X.columns:
                if cat in historical_df.columns:
                    # build categories from historical data (preserve training categories)
                    try:
                        cats = pd.Categorical(historical_df[cat]).categories
                        X[cat] = pd.Categorical(X[cat], categories=cats)
                    except Exception:
                        X[cat] = X[cat].astype('category')
                else:
                    X[cat] = X[cat].astype('category')

        # Predict cumulative net weight for this date_arrival
        pred = float(model.predict(X)[0])

        # collect debug info for the first predicted day for this rm_id
        if current_date == dates[0]:
            try:
                # serialize the X row as a dict of simple scalars
                x_row = {c: (str(X.iloc[0][c]) if pd.api.types.is_categorical_dtype(X[c]) else float(X.iloc[0][c]) if np.issubdtype(type(X.iloc[0][c]), np.number) else str(X.iloc[0][c])) for c in X.columns}
            except Exception:
                x_row = {c: str(X.iloc[0][c]) for c in X.columns}
            debug_rows.append({
                'rm_id': rm_id,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'initial_hist_values': ';'.join(str(v) for v in hist_values),
                'first_X': str(x_row),
                'first_pred': pred
            })

        # enforce non-negativity (clip negatives to zero)
        if pred < 0.0:
            pred = 0.0

        # enforce per-rm_id monotonicity: cumulative must not decrease
        if pred < prev_cum:
            pred = prev_cum

        # store prediction and update previous cumulative for next day
        date_preds[current_date.normalize()] = pred
        prev_cum = pred

        # append prediction to history and keep last 28
        hist_values.append(prev_cum)
        hist_values = hist_values[-28:]

        # update state so that downstream logic (and next-day features) see the new values
        state['cum_net_weight'] = prev_cum
        state['lag_1_day'] = float(hist_values[-1]) if len(hist_values) >= 1 else prev_cum
        state['lag_7_days'] = float(hist_values[-7]) if len(hist_values) >= 7 else float(hist_values[0])
        state['lag_14_days'] = float(hist_values[-14]) if len(hist_values) >= 14 else float(hist_values[0])
        state['lag_28_days'] = float(hist_values[-28]) if len(hist_values) >= 28 else float(hist_values[0])
        state['rolling_mean_7_days'] = safe_agg(hist_values, 7, lambda v: sum(v) / len(v))
        state['rolling_mean_14_days'] = safe_agg(hist_values, 14, lambda v: sum(v) / len(v))
        state['rolling_mean_28_days'] = safe_agg(hist_values, 28, lambda v: sum(v) / len(v))
        state['rolling_std_7_days'] = safe_std(hist_values[-7:])
        state['rolling_std_14_days'] = safe_std(hist_values[-14:])
        state['rolling_std_28_days'] = safe_std(hist_values[-28:])

    # Now map predictions back to requested IDs in group
    for _, row in group.iterrows():
        target_date = pd.to_datetime(row['forecast_end_date']).normalize()
        pred_value = date_preds.get(target_date, 0.0)
        results.append({'ID': row['ID'], 'predicted_weight': pred_value})

# Build submission DataFrame and save
submission = pd.DataFrame(results).sort_values('ID')
submission.to_csv('outputs/predictions_submission.csv', index=False)
print(f"Saved submission to outputs/predictions_submission.csv ({submission.shape[0]} rows)")

# Save debug CSV
debug_df = pd.DataFrame(debug_rows)
debug_path = Path('outputs') / 'prediction_debug.csv'
debug_df.to_csv(debug_path, index=False)
print(f"Saved prediction debug to: {debug_path} ({len(debug_df)} rows)")


