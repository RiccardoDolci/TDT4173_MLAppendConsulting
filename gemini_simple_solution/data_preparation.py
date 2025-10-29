#!/usr/bin/env python

"""
This script prepares the data for the TDT4173 machine learning task.

It works by:
1. Loading and cleaning historical receival data and purchase order data.
2. Creating a training set from historical data that mimics the cumulative prediction task.
3. Creating features for the test set (`prediction_mapping.csv`).
4. Saving the training data with targets to `training_dataset.csv`.
5. Saving the test features to `test_features.csv`.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import pathlib

# --- Get the directory where this script is located ---
BASE_DIR = pathlib.Path(__file__).parent.resolve()

print("Script started...")

# --- 1. Load and Prepare Data ---

def load_data():
    """
    Loads all required CSV files and performs initial date conversions.
    """
    print("Loading data...")
    print(f"Looking for files in: {BASE_DIR}")
    try:
        receivals_file = BASE_DIR / 'receivals.csv'
        po_file = BASE_DIR / 'purchase_orders.csv'
        materials_file = BASE_DIR / 'materials.csv'
        mapping_file = BASE_DIR / 'prediction_mapping.csv'
        
        receivals = pd.read_csv(receivals_file)
        purchase_orders = pd.read_csv(po_file)
        
        materials = pd.read_csv(materials_file)
        prediction_mapping = pd.read_csv(mapping_file)

        receivals['date_arrival'] = pd.to_datetime(
            receivals['date_arrival'], utc=True, errors='coerce'
        ).dt.date

        purchase_orders['delivery_date'] = pd.to_datetime(
            purchase_orders['delivery_date'], utc=True, errors='coerce'
        ).dt.date
        
        prediction_mapping['forecast_start_date'] = pd.to_datetime(
            prediction_mapping['forecast_start_date'], errors='coerce'
        ).dt.date
        prediction_mapping['forecast_end_date'] = pd.to_datetime(
            prediction_mapping['forecast_end_date'], errors='coerce'
        ).dt.date
        
        receivals = receivals.dropna(subset=['date_arrival', 'rm_id', 'net_weight'])
        purchase_orders = purchase_orders.dropna(subset=['delivery_date', 'product_id', 'quantity'])
        
        return receivals, purchase_orders, materials, prediction_mapping

    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}")
        print(f"The script was looking in this directory: {BASE_DIR}")
        print("Please make sure all CSV files are in the same folder as this script.")
        exit(1)

def aggregate_daily_data(receivals, purchase_orders, materials):
    """
    Aggregates receivals and purchase orders to a daily level per rm_id.
    """
    print("Aggregating daily data...")
    
    daily_receivals = receivals.groupby(
        ['rm_id', 'date_arrival']
    ).net_weight.sum().reset_index()
    
    rm_product_map = materials[['rm_id', 'product_id']].drop_duplicates().dropna()
    
    po_with_rm = pd.merge(
        purchase_orders,
        rm_product_map,
        on='product_id',
        how='left'
    ).dropna(subset=['rm_id'])
    
    daily_po = po_with_rm.groupby(
        ['rm_id', 'delivery_date']
    ).quantity.sum().reset_index()

    daily_receivals['rm_id'] = daily_receivals['rm_id'].astype(int)
    daily_po['rm_id'] = daily_po['rm_id'].astype(int)
    
    return daily_receivals, daily_po


# --- 2. Feature Engineering ---

def create_features_by_rm(df_rm, daily_receivals_rm, daily_po_rm):
    """
    Creates features for a single rm_id's data.
    """
    if df_rm.empty:
        return pd.DataFrame()

    features = df_rm.copy()
    features['original_index'] = features.index
    
    features['window_length'] = (
        features['forecast_end_date'] - features['forecast_start_date']
    ).apply(lambda x: x.days + 1)
    
    features['end_month'] = features['forecast_end_date'].apply(lambda x: x.month)
    
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
    
    feature_cols = [
        'rm_id', 'window_length', 'end_month',
        'po_in_window', 'hist_rec_30d', 'hist_po_30d'
    ]
    
    final_feature_cols = feature_cols + ['original_index', 'forecast_start_date', 'forecast_end_date']
    
    if 'original_index' in final_features.columns:
        final_features = final_features.set_index('original_index', drop=False)
    
    return final_features


# --- 3. Generate Training Data ---

def generate_training_data(daily_receivals, daily_po, years_to_use):
    """
    Generates a training DataFrame by creating cumulative windows from historical years.
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
        
        daily_receivals_rm = daily_receivals[daily_receivals['rm_id'] == rm_id]
        daily_po_rm = daily_po[daily_po['rm_id'] == rm_id]
        
        X_train_rm = create_features_by_rm(train_df_rm, daily_receivals_rm, daily_po_rm)
        
        if X_train_rm.empty:
            continue

        merged_targets = pd.merge(
            X_train_rm,
            daily_receivals_rm,
            on='rm_id',
            how='left'
        )
        
        target_mask = (merged_targets['date_arrival'] >= merged_targets['forecast_start_date']) & \
                      (merged_targets['date_arrival'] <= merged_targets['forecast_end_date'])
        
        y_agg = merged_targets[target_mask].groupby('original_index').net_weight.sum()
        
        X_train_rm = X_train_rm.set_index('original_index')
        
        y_train_rm = pd.Series(0, index=X_train_rm.index, name='target')
        y_train_rm.update(y_agg)
        
        X_train_list.append(X_train_rm)
        y_train_list.append(y_train_rm)

    print("Concatenating all training data...")
    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)
    
    feature_cols = [
        'rm_id', 'window_length', 'end_month',
        'po_in_window', 'hist_rec_30d', 'hist_po_30d'
    ]
    
    return X_train[feature_cols], y_train

# --- 4. Main Execution ---

def main():
    receivals, purchase_orders, materials, prediction_mapping = load_data()
    daily_receivals, daily_po = aggregate_daily_data(receivals, purchase_orders, materials)
    
    X_train, y_train = generate_training_data(
        daily_receivals, daily_po, years_to_use=[2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022, 2023, 2024]
    )

    print("Exporting training dataset...")
    training_data_to_export = X_train.copy()
    training_data_to_export['target'] = y_train
    training_data_filename = BASE_DIR / 'training_dataset.csv'
    training_data_to_export.to_csv(training_data_filename, index=True) 
    print(f"Successfully exported training data to {training_data_filename}")
    
    test_df_raw = prediction_mapping.set_index('ID')
    X_test = create_features(test_df_raw, daily_receivals, daily_po)
    
    feature_cols = X_train.columns.tolist()
    X_test_final = X_test[feature_cols]

    print("Exporting test features...")
    test_features_filename = BASE_DIR / 'test_features.csv'
    X_test_final.to_csv(test_features_filename, index=True)
    print(f"Successfully exported test features to {test_features_filename}")

    print("-" * 30)
    print("Data preparation complete.")
    print("-" * 30)

if __name__ == "__main__":
    main()
