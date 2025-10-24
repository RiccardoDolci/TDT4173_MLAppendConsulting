#!/usr/bin/env python

"""
This script generates predictions for the TDT4173 machine learning task.

It works by:
1. Loading and cleaning historical receival data and purchase order data.
2. Creating a training set from historical data (2022-2024) that
   mimics the cumulative prediction task.
3. Training a LightGBM regression model to predict the cumulative
   `net_weight` based on features like scheduled purchase order
   quantity in the window, historical receivals, and window length.
4. *** MODIFICATION: The model is now trained using 0.2 Quantile Loss ***
5. Using this model to predict the values required by
   `prediction_mapping.csv`.
6. Saving the result to `submission.csv`.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import pathlib  # Import pathlib to find script's location

# --- Get the directory where this script is located ---
# This will be used to find the CSV files.
BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Try to import LightGBM, provide install instructions if it fails
try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb # <-- Added for early stopping
    from sklearn.model_selection import train_test_split # <-- Added for validation
except ImportError:
    print("Error: lightgbm or scikit-learn package not found.")
    print("Please install them using: pip install lightgbm scikit-learn")
    exit(1)

print("Script started...")

# --- 1. Load and Prepare Data ---

def load_data():
    """
    Loads all required CSV files and performs initial date conversions.
    """
    print("Loading data...")
    print(f"Looking for files in: {BASE_DIR}")
    try:
        # --- Build full paths to the files ---
        receivals_file = BASE_DIR / 'receivals.csv'
        po_file = BASE_DIR / 'purchase_orders.csv'
        materials_file = BASE_DIR / 'materials.csv'
        mapping_file = BASE_DIR / 'prediction_mapping.csv'
        
        # Load main data using the full paths
        receivals = pd.read_csv(receivals_file)
        purchase_orders = pd.read_csv(po_file)
        
        # Load mapping/metadata
        materials = pd.read_csv(materials_file)
        prediction_mapping = pd.read_csv(mapping_file)

        # --- Parse dates ---
        # Receivals: Convert UTC timestamp to simple date
        receivals['date_arrival'] = pd.to_datetime(
            receivals['date_arrival'], utc=True, errors='coerce'
        ).dt.date

        # Purchase Orders: Convert timestamp to simple date
        purchase_orders['delivery_date'] = pd.to_datetime(
            purchase_orders['delivery_date'], utc=True, errors='coerce'
        ).dt.date
        
        # Prediction Mapping: Convert to simple date
        prediction_mapping['forecast_start_date'] = pd.to_datetime(
            prediction_mapping['forecast_start_date'], errors='coerce'
        ).dt.date
        prediction_mapping['forecast_end_date'] = pd.to_datetime(
            prediction_mapping['forecast_end_date'], errors='coerce'
        ).dt.date
        
        # Drop rows where date parsing failed
        receivals = receivals.dropna(subset=['date_arrival', 'rm_id', 'net_weight'])
        purchase_orders = purchase_orders.dropna(subset=['delivery_date', 'product_id', 'quantity'])
        
        return receivals, purchase_orders, materials, prediction_mapping

    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}")
        print(f"The script was looking in this directory: {BASE_DIR}")
        print("Please make sure all CSV files (receivals.csv, purchase_orders.csv,")
        print("materials.csv, prediction_mapping.csv) are in the same folder as this script.")
        exit(1)

def aggregate_daily_data(receivals, purchase_orders, materials):
    """
    Aggregates receivals and purchase orders to a daily level per rm_id.
    """
    print("Aggregating daily data...")
    
    # 1. Aggregate receivals
    daily_receivals = receivals.groupby(
        ['rm_id', 'date_arrival']
    ).net_weight.sum().reset_index()
    
    # 2. Create a map from product_id to rm_id using materials.csv
    rm_product_map = materials[['rm_id', 'product_id']].drop_duplicates().dropna()
    
    # 3. Link POs to rm_id
    po_with_rm = pd.merge(
        purchase_orders,
        rm_product_map,
        on='product_id',
        how='left'
    ).dropna(subset=['rm_id']) # Only keep POs we can map
    
    # 4. Aggregate purchase orders
    daily_po = po_with_rm.groupby(
        ['rm_id', 'delivery_date']
    ).quantity.sum().reset_index()

    # Ensure rm_id is a consistent integer type
    daily_receivals['rm_id'] = daily_receivals['rm_id'].astype(int)
    daily_po['rm_id'] = daily_po['rm_id'].astype(int)
    
    return daily_receivals, daily_po


# --- 2. Feature Engineering ---

def create_features(df, daily_receivals, daily_po):
    """
    Creates features for the given dataframe (train or test).
    'df' must have columns: rm_id, forecast_start_date, forecast_end_date
    """
    print(f"Creating features for {len(df)} rows...")
    
    # Use the index to uniquely identify each row after merges
    original_index_name = df.index.name
    features = df.reset_index()
    
    # Standardize the index column name
    if 'index' not in features.columns and original_index_name is not None:
        features = features.rename(columns={original_index_name: 'index'})

    
    # --- Date-based features ---
    features['window_length'] = (
        features['forecast_end_date'] - features['forecast_start_date']
    ).apply(lambda x: x.days + 1)
    
    features['end_month'] = features['forecast_end_date'].apply(lambda x: x.month)
    features['end_day_of_year'] = features['forecast_end_date'].apply(lambda x: x.timetuple().tm_yday)
    
    # --- PO in Window feature ---
    # Merge all POs onto the feature rows (many-to-many)
    po_merged = pd.merge(
        features,
        daily_po,
        on='rm_id',
        how='left'
    )
    
    # Filter for POs that fall inside the window for that row
    po_in_window_mask = (po_merged['delivery_date'] >= po_merged['forecast_start_date']) & \
                        (po_merged['delivery_date'] <= po_merged['forecast_end_date'])
    
    # Group by the original row index (now 'index') and sum the PO quantity
    po_in_window_agg = po_merged[po_in_window_mask].groupby('index').quantity.sum().rename('po_in_window')
    
    # Merge the aggregated feature back
    features = pd.merge(
        features,
        po_in_window_agg,
        left_on='index',
        right_index=True,
        how='left'
    ).fillna({'po_in_window': 0}) # Fill 0 for rows with no POs
    
    # --- Historical Features (relative to forecast_start_date) ---
    print("  Creating historical features...")
    hist_agg_list = []
    
    # Process one start_date at a time to avoid recomputing
    for start_date in features['forecast_start_date'].unique():
        ref_date = start_date - timedelta(days=1)
        hist_start_30d = ref_date - timedelta(days=29)

        # Filter data for the 30-day history window
        hist_rec_30d = daily_receivals[
            (daily_receivals['date_arrival'] >= hist_start_30d) &
            (daily_receivals['date_arrival'] <= ref_date)
        ]
        hist_po_30d = daily_po[
            (daily_po['delivery_date'] >= hist_start_30d) &
            (daily_po['delivery_date'] <= ref_date)
        ]
        
        # Aggregate by rm_id
        rec_agg = hist_rec_30d.groupby('rm_id').net_weight.sum().rename('hist_rec_30d')
        po_agg = hist_po_30d.groupby('rm_id').quantity.sum().rename('hist_po_30d')
        
        # Combine and store
        hist_agg = pd.concat([rec_agg, po_agg], axis=1).fillna(0).reset_index()
        hist_agg['forecast_start_date'] = start_date
        hist_agg_list.append(hist_agg)
        
    # Merge all historical features back to the main feature set
    if hist_agg_list:
        all_hist_agg = pd.concat(hist_agg_list, ignore_index=True)
        features = pd.merge(
            features,
            all_hist_agg,
            on=['rm_id', 'forecast_start_date'],
            how='left'
        ).fillna(0) # Fill 0 for new rm_ids with no history
    else:
        features['hist_rec_30d'] = 0
        features['hist_po_30d'] = 0

    # Define feature columns and set index back
    feature_cols = [
        'rm_id', 'window_length', 'end_month', 'end_day_of_year',
        'po_in_window', 'hist_rec_30d', 'hist_po_30d'
    ]
    
    # Add 'index' for target creation later
    final_feature_cols = feature_cols + ['index', 'forecast_start_date', 'forecast_end_date']
    
    return features[final_feature_cols].set_index('index')


# --- 3. Generate Training Data ---

def generate_training_data(daily_receivals, daily_po, years_to_use):
    """
    Generates a training DataFrame by creating cumulative windows
    from historical years.
    """
    print("Generating training data...")
    train_windows = []
    all_rms = daily_receivals['rm_id'].unique()
    
    for rm in all_rms:
        for year in years_to_use:
            start_date = date(year, 1, 1)
            # Max window is 151 days (Jan 1 to May 31)
            for days in range(1, 152): 
                end_date = start_date + timedelta(days=days)
                
                # Stop if we cross into the prediction period
                if end_date >= date(2025, 1, 1):
                    break
                    
                train_windows.append({
                    'rm_id': rm,
                    'forecast_start_date': start_date,
                    'forecast_end_date': end_date
                })
                
    train_df_raw = pd.DataFrame(train_windows)
    
    # Create features
    X_train = create_features(train_df_raw, daily_receivals, daily_po)
    
    # --- Create targets (y_train) ---
    print("  Creating training targets...")
    # Merge raw windows with all receivals
    merged_targets = pd.merge(
        X_train.reset_index(),
        daily_receivals,
        on='rm_id',
        how='left'
    )
    
    # Filter for receivals that fall inside the window
    target_mask = (merged_targets['date_arrival'] >= merged_targets['forecast_start_date']) & \
                  (merged_targets['date_arrival'] <= merged_targets['forecast_end_date'])
    
    # Group by the original index and sum
    y_agg = merged_targets[target_mask].groupby('index').net_weight.sum()
    
    # Create the final y_train Series, aligning with X_train's index
    y_train = pd.Series(0, index=X_train.index, name='target')
    y_train.update(y_agg) # Update with the summed values
    
    # Drop helper columns from X_train
    feature_cols = [
        'rm_id', 'window_length', 'end_month', 'end_day_of_year',
        'po_in_window', 'hist_rec_30d', 'hist_po_30d'
    ]
    
    return X_train[feature_cols], y_train

# --- 4. Main Execution ---

def main():
    # 1. Load and aggregate
    receivals, purchase_orders, materials, prediction_mapping = load_data()
    daily_receivals, daily_po = aggregate_daily_data(receivals, purchase_orders, materials)
    
    # 2. Generate Training Data (use 2022, 2023, 2024 for training)
    X_train, y_train = generate_training_data(
        daily_receivals, daily_po, years_to_use=[2022, 2023, 2024]
    )

    # --- NEW: Export the training dataset ---
    print("Exporting training dataset...")
    training_data_to_export = X_train.copy()
    training_data_to_export['target'] = y_train
    training_data_filename = BASE_DIR / 'training_dataset.csv'
    # Save with index because the index is used in the feature/target generation
    training_data_to_export.to_csv(training_data_filename, index=True) 
    print(f"Successfully exported training data to {training_data_filename}")
    # --- END NEW ---
    
    # 3. Create Test Data
    # Set 'ID' as the index to track rows
    test_df_raw = prediction_mapping.set_index('ID')
    X_test = create_features(test_df_raw, daily_receivals, daily_po)
    
    # Keep only final feature columns
    feature_cols = X_train.columns.tolist()
    X_test = X_test[feature_cols]

    # 4. Train Model
    print("Training model...")
    
    # --- MODIFICATION: Create validation set ---
    X_train_full, X_val, y_train_full, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    # Convert rm_id to 'category' dtype for LightGBM
    X_train_full['rm_id'] = X_train_full['rm_id'].astype('category')
    X_val['rm_id'] = pd.Categorical(
        X_val['rm_id'],
        categories=X_train_full['rm_id'].cat.categories
    )
    
    # Align test categories with train categories
    X_test['rm_id'] = pd.Categorical(
        X_test['rm_id'],
        categories=X_train_full['rm_id'].cat.categories
    )

    # --- MODIFICATION: Set objective to quantile ---
    model = LGBMRegressor(
        objective='quantile',  # <-- Use quantile loss
        alpha=0.2,             # <-- Set to 0.2 quantile
        random_state=42,
        n_estimators=1000,     # <-- Increase n_estimators for early stopping
        learning_rate=0.05,    # <-- Lower learning rate
        n_jobs=-1,
        categorical_feature=['rm_id']
    )
    
    print("Starting model fitting...")
    # --- MODIFICATION: Use early stopping ---
    model.fit(
        X_train_full, y_train_full,
        eval_set=[(X_val, y_val)],
        eval_metric='quantile', # <-- Evaluate on the correct metric
        callbacks=[lgb.early_stopping(100, verbose=True)]
    )
    print("Model training complete.")

    # 5. Generate Predictions
    print("Generating predictions...")
    predictions = model.predict(X_test)
    
    # We can't have negative weight
    predictions[predictions < 0] = 0
    
    # 6. Save Submission
    submission = pd.DataFrame({
        'ID': X_test.index,
        'predicted_weight': predictions
    })
    
    # Save the submission file in the same directory as the script
    submission_filename = BASE_DIR / 'submission.csv'
    submission.to_csv(submission_filename, index=False)
    
    print("-" * 30)
    print(f"Success! Predictions saved to {submission_filename}")
    print(f"Submission file has {len(submission)} rows (expected 30450).")
    print(submission.head())
    print("-" * 30)

if __name__ == "__main__":
    main()