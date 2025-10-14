# Import required libraries
import pandas as pd
import numpy as np
from itertools import product

print("--- Loading Data ---")
df = pd.read_csv('data/mod_data/merged_clean_data.csv', sep=',')

rm_ids = df['rm_id'].unique()
# ------------------------------------------------------------------
# Create dataset: one row per (rm_id, date) for 2019-01-01 .. 2024-12-31
# with yearly-reset cumulative net_weight per rm_id (inclusive of date)
# We could include also supplier_id if needed
# We can also do it till 31 May for each year because prediction will be till then
# ------------------------------------------------------------------
print("\n--- Building yearly cumulative net_weight dataset (2019-01-01 to 2024-12-31) ---")
# date range requested by user (timezone-aware UTC to match source datetimes)
dates_req = pd.date_range(start='2019-01-01', end='2024-12-31', freq='D', tz='UTC')

# Build master grid for rm_id x requested dates
all_combinations_req = product(rm_ids, dates_req)
cum_master = pd.DataFrame(all_combinations_req, columns=['rm_id', 'date'])
# ensure 'date' column is timezone-aware (should already be from dates_req)
cum_master['date'] = pd.to_datetime(cum_master['date'], utc=True)

# Ensure source df has datetime-normalized dates and net_weight
df_net = df.copy()
df_net['date_arrival'] = pd.to_datetime(df_net['date_arrival'], errors='coerce')
df_net['date'] = df_net['date_arrival'].dt.normalize()

# Aggregate net_weight per rm_id per date (sum if multiple receivals same day)
daily_net = (
    df_net.groupby(['rm_id', 'date'], as_index=False)['net_weight']
    .sum()
)

# Merge master grid with actual daily net weights (missing -> 0)
final_df = pd.merge(cum_master, daily_net, on=['rm_id', 'date'], how='left')
final_df['net_weight'] = final_df['net_weight'].fillna(0)

# Year column for yearly reset
final_df['year'] = final_df['date'].dt.year

# Sort and compute yearly cumulative sum per rm_id
final_df = final_df.sort_values(by=['rm_id', 'date'])
final_df['cum_net_weight'] = final_df.groupby(['rm_id', 'year'])['net_weight'].cumsum()

# Keep only requested columns
cum_out = final_df[['rm_id', 'date', 'cum_net_weight']]

# Save result
# out_path_cum = 'data/mod_data/rm_yearly_cum_netweight_2019_2024.csv'
# cum_out.to_csv(out_path_cum, index=False)
# print(f"Wrote yearly cumulative dataset to {out_path_cum} with shape {cum_out.shape}")
# ------------------------------------------------------------------

print("\n--- Engineering Date Features ---")
final_df['year'] = final_df['date'].dt.year
final_df['month'] = final_df['date'].dt.month
final_df['day'] = final_df['date'].dt.day
final_df['ID'] = final_df['date'].dt.day-1
final_df['day_of_week'] = final_df['date'].dt.dayofweek
final_df['week_of_year'] = final_df['date'].dt.isocalendar().week

# --- Engineer a Closure/Holiday Feature ---
print("Engineering closure/holiday feature...")
# Calculate the total net weight delivered per day across all materials
daily_total_weight = final_df.groupby('date')['net_weight'].sum()
# Identify days where the total delivery was zero
closure_dates = daily_total_weight[daily_total_weight == 0].index
# Create the new binary feature
final_df['is_closure_day'] = final_df['date'].isin(closure_dates).astype(int)
print(f"Identified {len(closure_dates)} potential closure/holiday dates.")
# --- End of New Section ---

print("\n--- Engineering Lag and Rolling Window Features ---")
final_df = final_df.sort_values(by=['rm_id', 'date'])

final_df['lag_1_day'] = final_df.groupby('rm_id')['cum_net_weight'].shift(1)
final_df['lag_7_days'] = final_df.groupby('rm_id')['cum_net_weight'].shift(7)
final_df['lag_14_days'] = final_df.groupby('rm_id')['cum_net_weight'].shift(14)
final_df['lag_28_days'] = final_df.groupby('rm_id')['cum_net_weight'].shift(28)

windows = [7, 14, 28]
for window in windows:
    final_df[f'rolling_mean_{window}_days'] = final_df.groupby('rm_id')['cum_net_weight'].shift(1).rolling(window=window).mean()
    final_df[f'rolling_std_{window}_days'] = final_df.groupby('rm_id')['cum_net_weight'].shift(1).rolling(window=window).std()

final_df.fillna(0, inplace=True)

# Time Since Last Delivery
final_df['last_delivery_date'] = final_df['date'].where(final_df['net_weight'] > 0)
final_df['last_delivery_date'] = final_df.groupby('rm_id')['last_delivery_date'].ffill()
final_df['time_since_last_delivery'] = (final_df['date'] - final_df['last_delivery_date']).dt.days

# Correctly fill and cap the feature
final_df['time_since_last_delivery'].fillna(999, inplace=True)
final_df['time_since_last_delivery'] = final_df['time_since_last_delivery'].clip(upper=999)
final_df = final_df.drop(columns=['last_delivery_date'])
print("All features created successfully.")

print("\n--- Finalizing and Saving Dataset ---")
# We keep out 'time_since_last_delivery' for now, can be added later if needed
# We keep out 'year' and 'date' as well since it's not needed for modeling
features_to_keep = [
    'ID','rm_id', 'cum_net_weight', 'day', 'month',
    'day_of_week', 'week_of_year', 'is_closure_day',
    'lag_1_day', 'lag_7_days', 'lag_14_days', 'lag_28_days',
    'rolling_mean_7_days', 'rolling_mean_14_days', 'rolling_mean_28_days',
    'rolling_std_7_days', 'rolling_std_14_days', 'rolling_std_28_days', 
]

model_ready_df = final_df[features_to_keep]

# Save to Feather format for speed and type preservation
output_path = 'data/mod_data/model_ready_data.csv'
model_ready_df.to_csv(output_path, index=False)

print("Final dataset created successfully!")
print(model_ready_df.head())
print(f"\nDataset saved to '{output_path}'")
