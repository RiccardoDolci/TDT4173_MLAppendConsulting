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

def create_features_by_rm(df_rm, daily_receivals_rm, daily_po_rm):
    """
    Creates features for a single rm_id's data.
    'df_rm' must have columns: rm_id, forecast_start_date, forecast_end_date
    """
    if df_rm.empty:
        return pd.DataFrame()

    features = df_rm.copy()
    features['original_index'] = features.index
    
    # --- Date-based features ---
    features['window_length'] = (
        features['forecast_end_date'] - features['forecast_start_date']
    ).apply(lambda x: x.days + 1)
    
    features['end_month'] = features['forecast_end_date'].apply(lambda x: x.month)
    
    # --- PO in Window feature ---
    if not daily_po_rm.empty:
        po_merged = pd.merge(
            features,
            daily_po_rm,
            on='rm_id',
            how='left'
        )
        
        po_in_window_mask = (po_merged['delivery_date'] >= po_merged['forecast_start_date']) & \
                            (po_merged['delivery_date'] <= po_merged['forecast_end_date'])
        
        po_in_window_agg = po_merged[po_in_window_mask].groupby('original_index').quantity.sum().rename('po_in_window')
        
        features = pd.merge(
            features,
            po_in_window_agg,
            left_index=True,
            right_index=True,
            how='left'
        ).fillna({'po_in_window': 0})
    else:
        features['po_in_window'] = 0
        
    # --- Historical Features (relative to forecast_start_date) ---
    hist_agg_list = []
    
    for start_date in features['forecast_start_date'].unique():
        ref_date = start_date - timedelta(days=1)
        hist_start_30d = ref_date - timedelta(days=29)

        hist_rec_30d = daily_receivals_rm[
            (daily_receivals_rm['date_arrival'] >= hist_start_30d) &
            (daily_receivals_rm['date_arrival'] <= ref_date)
        ]
        hist_po_30d = daily_po_rm[
            (daily_po_rm['delivery_date'] >= hist_start_30d) &
            (daily_po_rm['delivery_date'] <= ref_date)
        ]
        
        rec_agg_val = hist_rec_30d.net_weight.sum()
        po_agg_val = hist_po_30d.quantity.sum()
        
        hist_agg_list.append({
            'forecast_start_date': start_date,
            'hist_rec_30d': rec_agg_val,
            'hist_po_30d': po_agg_val
        })
        
    if hist_agg_list:
        all_hist_agg = pd.DataFrame(hist_agg_list)
        features = pd.merge(
            features,
            all_hist_agg,
            on='forecast_start_date',
            how='left'
        ).fillna(0)
    else:
        features['hist_rec_30d'] = 0
        features['hist_po_30d'] = 0

    feature_cols = [
        'rm_id', 'window_length', 'end_month',
        'po_in_window', 'hist_rec_30d', 'hist_po_30d'
    ]
    
    final_feature_cols = feature_cols + ['original_index', 'forecast_start_date', 'forecast_end_date']
    
    return features[final_feature_cols]

def create_features(df, daily_receivals, daily_po):
    """
    Creates features for the given dataframe by processing one rm_id at a time.
    """
    print(f"Creating features for {len(df)} rows...")
    
    all_rms = df['rm_id'].unique()
    all_features = []

    for i, rm_id in enumerate(all_rms):
        if (i + 1) % 50 == 0:
            print(f"  Processing rm_id {i+1}/{len(all_rms)}")
            
        df_rm = df[df['rm_id'] == rm_id]
        daily_receivals_rm = daily_receivals[daily_receivals['rm_id'] == rm_id]
        daily_po_rm = daily_po[daily_po['rm_id'] == rm_id]
        
        features_rm = create_features_by_rm(df_rm, daily_receivals_rm, daily_po_rm)
        all_features.append(features_rm)

    if not all_features:
        return pd.DataFrame()

    final_features = pd.concat(all_features)
    
    # Define feature columns and set index back
    feature_cols = [
        'rm_id', 'window_length', 'end_month',
        'po_in_window', 'hist_rec_30d', 'hist_po_30d'
    ]
    
    # Add original index for target creation later
    final_feature_cols = feature_cols + ['original_index', 'forecast_start_date', 'forecast_end_date']
    
    # Keep original index for target alignment
    if 'original_index' in final_features.columns:
        final_features = final_features.set_index('original_index', drop=False)
    
    return final_features


# --- 3. Generate Training Data ---

def generate_training_data(daily_receivals, daily_po, years_to_use):
    """
    Generates a training DataFrame by creating cumulative windows
    from historical years, processing one rm_id at a time to save memory.
    """
    print("Generating training data...")
    
    all_rms = daily_receivals['rm_id'].unique()
    X_train_list = []
    y_train_list = []

    for i, rm_id in enumerate(all_rms):
        if (i + 1) % 50 == 0:
            print(f"  Generating training data for rm_id {i+1}/{len(all_rms)}")

        train_windows_rm = []
        for year in years_to_use:
            start_date = date(year, 1, 1)
            for days in range(1, 152): 
                end_date = start_date + timedelta(days=days)
                if end_date >= date(2025, 1, 1):
                    break
                train_windows_rm.append({
                    'rm_id': rm_id,
                    'forecast_start_date': start_date,
                    'forecast_end_date': end_date
                })
        
        if not train_windows_rm:
            continue
            
        train_df_rm = pd.DataFrame(train_windows_rm)
        
        # --- Create features for this rm_id ---
        daily_receivals_rm = daily_receivals[daily_receivals['rm_id'] == rm_id]
        daily_po_rm = daily_po[daily_po['rm_id'] == rm_id]
        
        X_train_rm = create_features_by_rm(train_df_rm, daily_receivals_rm, daily_po_rm)
        
        if X_train_rm.empty:
            continue

        # --- Create targets for this rm_id ---
        merged_targets = pd.merge(
            X_train_rm,
            daily_receivals_rm,
            on='rm_id',
            how='left'
        )
        
        target_mask = (merged_targets['date_arrival'] >= merged_targets['forecast_start_date']) & \
                      (merged_targets['date_arrival'] <= merged_targets['forecast_end_date'])
        
        y_agg = merged_targets[target_mask].groupby('original_index').net_weight.sum()
        
        # The index of X_train_rm should be original_index to align with y_agg
        X_train_rm = X_train_rm.set_index('original_index')
        
        y_train_rm = pd.Series(0, index=X_train_rm.index, name='target')
        y_train_rm.update(y_agg)
        
        # Append to lists
        X_train_list.append(X_train_rm)
        y_train_list.append(y_train_rm)

    # Concatenate all parts at the end
    print("Concatenating all training data...")
    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)
    
    # Drop helper columns from X_train
    feature_cols = [
        'rm_id', 'window_length', 'end_month',
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
        daily_receivals, daily_po, years_to_use=[2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022, 2023, 2024]
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