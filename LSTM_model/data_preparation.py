import pandas as pd
import numpy as np
from datetime import date, timedelta
import pathlib  # Import pathlib to find script's location
import os
import json
import joblib
#
# LSTM_model data preparation and feature generation
#
# This module encapsulates data-loading, cleaning, aggregation and feature
# engineering required to build the model-ready dataset used by
# ``LSTM_model/code_lstm.py``.
#
# Public API
# - load_data() -> (receivals, purchase_orders, materials, prediction_mapping)
# - aggregate_daily_data(receivals, purchase_orders, materials) -> (daily_receivals, daily_po)
# - create_features(df, daily_receivals, daily_po) -> features DataFrame (indexed by original row index)
# - generate_training_data(daily_receivals, daily_po, years_to_use) -> (X_train, y_train)
# - main() -> writes ``data/training_dataset.csv`` and returns its path
#
# Inputs (expected files in ``LSTM_model/data/``):
# - receivals.csv, purchase_orders.csv, materials.csv, prediction_mapping.csv
#
# Outputs:
# - data/training_dataset.csv (model-ready features + target)
#
# Notes:
# - The code converts quantities in pounds (unit_id==43) to kilograms.
# - Date columns are parsed with coercion; rows with invalid key fields are dropped.
#
# --- Get the directory where this script is located ---
# This will be used to find the CSV files.
BASE_DIR = pathlib.Path(__file__).parent.resolve()
# --- 1. Load and Prepare Data ---

def load_data():
    """
    Loads all required CSV files and performs initial date conversions.
    """
    print("Loading data...")
    print(f"Looking for files in: {BASE_DIR}")
    try:
        # --- Build full paths to the files ---
        receivals_file = BASE_DIR / 'data/receivals.csv'
        po_file = BASE_DIR / 'data/purchase_orders.csv'
        materials_file = BASE_DIR / 'data/materials.csv'
        mapping_file = BASE_DIR / 'data/prediction_mapping.csv'

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

        # Remove rows with non-positive quantities/weights which are invalid for training
        # (zero or negative receivals/purchase order quantities likely indicate bad or placeholder data)
        receivals = receivals.loc[receivals['net_weight'] > 0].copy()
        purchase_orders = purchase_orders.loc[purchase_orders['quantity'] > 0].copy()

        # Define the conversion factor for LBS to KG
        LBS_TO_KG = 0.453592

        # Create a mask to identify rows with pounds (unit_id 43)
        pounds_mask = (purchase_orders['unit_id'] == 43)
        print(f"Found {pounds_mask.sum()} rows with pounds to convert.")

        # Apply the conversion to the 'quantity' column for these rows
        purchase_orders.loc[pounds_mask, 'quantity'] = purchase_orders.loc[pounds_mask, 'quantity'] * LBS_TO_KG

        # Update the unit and unit_id columns for consistency ---

        kg_unit_id = 40
        purchase_orders.loc[pounds_mask, 'unit'] = 'kg'
        purchase_orders.loc[pounds_mask, 'unit_id'] = kg_unit_id

        print("Pounds have been successfully converted to KG.")

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

    # Note: supplier-level information was removed — supplier indicator features
    # are not used any more. This function returns only daily_receivals and
    # daily_po to keep the API simple and memory-efficient.

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
    # --- End-week feature ---
    # end_week is the ordinal number of the weekend in the month that contains
    # the forecast_end_date. We define weekends as blocks starting on Saturday.
    # Algorithm (vectorized via apply): find the first Saturday of the month
    # and compute 1 + floor((date - first_saturday)/7). Dates before the first
    # Saturday are assigned to week 1.
    def _end_week(d):
        if pd.isna(d):
            return 0
        # d is a datetime.date; convert to datetime
        first_of_month = d.replace(day=1)
        # weekday(): Monday=0 .. Sunday=6; Saturday is 5
        days_until_sat = (5 - first_of_month.weekday()) % 7
        first_sat = first_of_month + timedelta(days=days_until_sat)
        if d < first_sat:
            return 1
        return 1 + ((d - first_sat).days // 7)

    #features['end_week'] = features['forecast_end_date'].apply(_end_week)
    #features['end_day_of_year'] = features['forecast_end_date'].apply(lambda x: x.timetuple().tm_yday)
    
    # --- PO in Window feature (memory-efficient per-rm computation) ---
    # Compute the sum of PO quantities whose delivery_date falls inside each
    # feature row's [forecast_start_date, forecast_end_date] window without
    # performing a global many-to-many merge which can explode memory for
    # large training grids.
    print("  Computing po_in_window feature (per-rm)...")
    pp = daily_po.copy()
    pp['delivery_date'] = pd.to_datetime(pp['delivery_date'], errors='coerce')

    po_by_rm = {rm: grp.sort_values('delivery_date').reset_index(drop=True)
                for rm, grp in pp.groupby('rm_id')} if not pp.empty else {}

    # prepare output array aligned with current features positional index
    po_in_window_vals = np.zeros(len(features), dtype=float)

    # features currently has a RangeIndex (from reset_index earlier) so rows.index
    # gives positional positions we can use to assign into po_in_window_vals.
    for rm, rows in features.groupby('rm_id'):
        row_positions = rows.index.values

        po_grp = po_by_rm.get(rm)
        if po_grp is None or po_grp.empty:
            continue

        po_dates = pd.to_datetime(po_grp['delivery_date']).values.astype('datetime64[D]')
        po_qty = po_grp['quantity'].values.astype(float)
        po_cum = np.cumsum(po_qty)

        starts = pd.to_datetime(rows['forecast_start_date']).values.astype('datetime64[D]')
        ends = pd.to_datetime(rows['forecast_end_date']).values.astype('datetime64[D]')

        left_pos = np.searchsorted(po_dates, starts, side='left')
        right_pos = np.searchsorted(po_dates, ends, side='right') - 1

        sums = np.zeros(len(row_positions), dtype=float)
        for i, (l, r) in enumerate(zip(left_pos, right_pos)):
            if r >= l and l < len(po_cum):
                left_cum = po_cum[l-1] if l > 0 else 0.0
                sums[i] = po_cum[r] - left_cum
            else:
                sums[i] = 0.0

        po_in_window_vals[row_positions] = sums

    features['po_in_window'] = po_in_window_vals

    # Supplier indicator features were removed (they caused high dimensionality
    # and memory issues). The feature set focuses on numeric aggregates instead.

    # --- Last year weight feature (memory-efficient per-rm computation) ---
    # Compute receivals sum in the same window one year earlier without
    # performing a global many-to-many merge which can blow memory.
    print("  Computing last_year_weight feature (per-rm, vectorized)...")

    # Prepare output array
    n_rows = len(features)
    last_year_vals = np.zeros(n_rows, dtype=float)

    # Build per-rm sorted receival arrays
    dr = daily_receivals.copy()
    # ensure datetime64[D]
    dr['date_arrival'] = pd.to_datetime(dr['date_arrival'], errors='coerce')
    rec_by_rm = {rm: grp.sort_values('date_arrival').reset_index(drop=True)
                 for rm, grp in dr.groupby('rm_id')}

    # Work on positional indices of the current features DataFrame
    features = features.reset_index(drop=True)

    # For each rm_id in the feature set, compute shifted-window sums
    for rm, rows in features.groupby('rm_id'):
        rows_pos = rows.index.values  # positions into features / last_year_vals

        rec = rec_by_rm.get(rm)
        if rec is None or rec.empty:
            continue

        rec_dates = pd.to_datetime(rec['date_arrival']).values.astype('datetime64[D]')
        rec_weights = rec['net_weight'].values.astype(float)
        rec_cum = np.cumsum(rec_weights)

        # compute shifted window boundaries
        starts = (pd.to_datetime(rows['forecast_start_date']) - pd.Timedelta(days=365)).values.astype('datetime64[D]')
        ends = (pd.to_datetime(rows['forecast_end_date']) - pd.Timedelta(days=365)).values.astype('datetime64[D]')

        left_pos = np.searchsorted(rec_dates, starts, side='left')
        right_pos = np.searchsorted(rec_dates, ends, side='right') - 1

        # compute sums
        sums = np.zeros(len(rows_pos), dtype=float)
        for i, (l, r) in enumerate(zip(left_pos, right_pos)):
            if r >= l and r >= 0 and l < len(rec_cum):
                left_cum = rec_cum[l-1] if l > 0 else 0.0
                sums[i] = rec_cum[r] - left_cum
            else:
                sums[i] = 0.0

        last_year_vals[rows_pos] = sums

    #features['last_year_weight'] = last_year_vals
    
    # --- Historical Features (relative to forecast_start_date) ---
    print("  Creating historical features (7/14/30/90-day)...")
    hist_agg_list = []

    # horizons to compute (days)
    rec_horizons = [30]
    po_horizons = [30]

    # Process one start_date at a time to avoid recomputing large filters
    for start_date in features['forecast_start_date'].unique():
        ref_date = start_date - timedelta(days=1)

        # containers for aggregated columns per rm_id
        rec_frames = []
        po_frames = []

        # Compute receivals aggregates for each requested horizon
        for h in rec_horizons:
            hist_start = ref_date - timedelta(days=(h - 1))
            hist_rec = daily_receivals[
                (daily_receivals['date_arrival'] >= hist_start) &
                (daily_receivals['date_arrival'] <= ref_date)
            ]
            rec_agg = hist_rec.groupby('rm_id').net_weight.sum().rename(f'hist_rec_{h}d')
            rec_frames.append(rec_agg)

        # Compute PO aggregates for each requested horizon
        for h in po_horizons:
            hist_start = ref_date - timedelta(days=(h - 1))
            hist_po = daily_po[
                (daily_po['delivery_date'] >= hist_start) &
                (daily_po['delivery_date'] <= ref_date)
            ]
            po_agg = hist_po.groupby('rm_id').quantity.sum().rename(f'hist_po_{h}d')
            po_frames.append(po_agg)

        # Combine all rec and po aggregates into a single DataFrame
        if rec_frames or po_frames:
            all_aggs = pd.concat(rec_frames + po_frames, axis=1)
            all_aggs = all_aggs.fillna(0).reset_index()
        else:
            all_aggs = pd.DataFrame(columns=['rm_id'])

        all_aggs['forecast_start_date'] = start_date
        hist_agg_list.append(all_aggs)

    # Merge all historical features back to the main feature set
    if hist_agg_list:
        all_hist_agg = pd.concat(hist_agg_list, ignore_index=True)
        features = pd.merge(
            features,
            all_hist_agg,
            on=['rm_id', 'forecast_start_date'],
            how='left'
        ).fillna(0)  # Fill 0 for new rm_ids with no history
    else:
        # Ensure the new columns exist even if no history is present
        for h in rec_horizons:
            features[f'hist_rec_{h}d'] = 0
        for h in po_horizons:
            features[f'hist_po_{h}d'] = 0

    # Define feature columns and include any supplier indicator columns that were
    # dynamically created above (they have names like 'supplier_<id>').
    core_feature_cols = [
        'rm_id', 'window_length', 'end_month',
        'po_in_window'
        # receival history
        # 'hist_rec_7d', 'hist_rec_14d'
        , 'hist_rec_30d'
        # , 'hist_rec_90d',
        # purchase order history
        # 'hist_po_7d', 'hist_po_14d'
        , 'hist_po_30d'
        # , 'hist_po_90d',
        # 'last_year_weight'
    ]

    feature_cols = core_feature_cols

    # Add 'index' for target creation later and keep forecast dates for downstream use
    final_feature_cols = feature_cols + ['index', 'forecast_start_date', 'forecast_end_date']

    return features[final_feature_cols].set_index('index')

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
    # Memory-efficient approach: compute target per rm_id using numpy arrays and
    # searchsorted on the per-rm sorted receival dates. This avoids the large
    # many-to-many merge that can explode memory for big training grids.
    print("  Creating training targets (memory-efficient per-rm computation)...")

    # Prepare output series
    y_train = pd.Series(0, index=X_train.index, name='target')

    # Convert dates in daily_receivals to plain datetime.date for safe comparisons
    dr = daily_receivals.copy()
    if dr['date_arrival'].dtype != 'O' and not np.issubdtype(dr['date_arrival'].dtype, np.datetime64):
        dr['date_arrival'] = pd.to_datetime(dr['date_arrival'], errors='coerce')

    # Group receivals by rm_id for efficient per-rm processing
    rec_by_rm = {rm: grp.sort_values('date_arrival').reset_index(drop=True)
                 for rm, grp in dr.groupby('rm_id')}

    # Iterate over X_train grouped by rm_id to compute sums for each window
    for rm, rows in X_train.reset_index().groupby('rm_id'):
        rows = rows.copy()
        rows_index = rows['index'].values

        rec = rec_by_rm.get(rm)
        if rec is None or rec.empty:
            # no receivals for this rm -> targets remain zero
            continue

        # numpy arrays for quick search and cumsum
        rec_dates = pd.to_datetime(rec['date_arrival']).values.astype('datetime64[D]')
        rec_weights = rec['net_weight'].values.astype(float)
        rec_cum = np.cumsum(rec_weights)

        # For each row/window, find left/right positions and compute window sum
        starts = pd.to_datetime(rows['forecast_start_date']).values.astype('datetime64[D]')
        ends = pd.to_datetime(rows['forecast_end_date']).values.astype('datetime64[D]')

        # use numpy.searchsorted (vectorized) to find positions
        left_pos = np.searchsorted(rec_dates, starts, side='left')
        right_pos = np.searchsorted(rec_dates, ends, side='right') - 1

        # compute sums safely
        sums = np.zeros(len(rows_index), dtype=float)
        for i, (l, r) in enumerate(zip(left_pos, right_pos)):
            if r >= l and r >= 0 and l < len(rec_cum):
                left_cum = rec_cum[l-1] if l > 0 else 0.0
                sums[i] = rec_cum[r] - left_cum
            else:
                sums[i] = 0.0

        # assign computed sums back to y_train using the original index
        y_train.loc[rows_index] = sums
    
    # Ensure the training matrix contains the core feature columns; return full X_train
    core_feature_cols = [
        'rm_id', 'window_length', 'end_month',
        'po_in_window',
        # receival history
        # 'hist_rec_7d', 'hist_rec_14d'
         'hist_rec_30d'
        # , 'hist_rec_90d',
        # purchase order history
        # 'hist_po_7d', 'hist_po_14d'
        , 'hist_po_30d'
        # , 'hist_po_90d',
        # 'last_year_weight'
    ]

    # If supplier columns exist, they will already be present in X_train; for safety ensure core columns exist
    missing = [c for c in core_feature_cols if c not in X_train.columns]
    if missing:
        raise RuntimeError(f"Missing expected feature columns after create_features: {missing}")

    # Drop helper/date columns before returning features for training. These
    # are useful internally for target creation but LightGBM requires numeric
    # dtypes only.
    X_train_export = X_train.drop(columns=['forecast_start_date', 'forecast_end_date'], errors='ignore')

    return X_train_export, y_train


def main():
    # 1. Load and aggregate
    receivals, purchase_orders, materials, prediction_mapping = load_data()
    daily_receivals, daily_po = aggregate_daily_data(receivals, purchase_orders, materials)

    # 2. Generate Training Data
    X_train, y_train = generate_training_data(
        daily_receivals, daily_po, years_to_use=[2012,2013,2014,2015,2016,2017,2018,2019, 2020, 2021, 2022, 2023, 2024]
    )
    # 3.Export the training dataset ---
    print("Exporting training dataset...")
    training_data_to_export = X_train.copy()
    training_data_to_export['target'] = y_train
    training_data_filename = BASE_DIR / 'data/training_dataset.csv'
    # Save with index because the index is used in the feature/target generation
    training_data_to_export.to_csv(training_data_filename, index=True) 
    print(f"Successfully exported training data to {training_data_filename}")
    # Return the path to the generated training dataset for programmatic use
    return training_data_filename

if __name__ == "__main__":
    main()