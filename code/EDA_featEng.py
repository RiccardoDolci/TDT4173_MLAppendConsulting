# Import required libraries
import pandas as pd
import numpy as np
from itertools import product

print("--- Loading and Merging Data ---")
df = pd.read_csv('data/mod_data/merged_clean_data.csv', sep=';')

df['date_arrival'] = pd.to_datetime(df['date_arrival']).dt.date

rm_ids = df['rm_id'].unique()
dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='D')

all_combinations = product(rm_ids, dates)
master_df = pd.DataFrame(all_combinations, columns=['rm_id', 'date_arrival'])
master_df['date_arrival'] = pd.to_datetime(master_df['date_arrival']).dt.date

final_df = pd.merge(master_df, df, on=['rm_id', 'date_arrival'], how='left')
final_df['net_weight'] = final_df['net_weight'].fillna(0)
final_df['date_arrival'] = pd.to_datetime(final_df['date_arrival'])
print("Base daily dataframe created.")

print("\n--- Engineering Date Features ---")
final_df['year'] = final_df['date_arrival'].dt.year
final_df['month'] = final_df['date_arrival'].dt.month
final_df['day'] = final_df['date_arrival'].dt.day
final_df['day_of_week'] = final_df['date_arrival'].dt.dayofweek
final_df['week_of_year'] = final_df['date_arrival'].dt.isocalendar().week

# --- Engineer a Closure/Holiday Feature ---
print("Engineering closure/holiday feature...")
# Calculate the total net weight delivered per day across all materials
daily_total_weight = final_df.groupby('date_arrival')['net_weight'].sum()
# Identify days where the total delivery was zero
closure_dates = daily_total_weight[daily_total_weight == 0].index
# Create the new binary feature
final_df['is_closure_day'] = final_df['date_arrival'].isin(closure_dates).astype(int)
print(f"Identified {len(closure_dates)} potential closure/holiday dates.")
# --- End of New Section ---

print("\n--- Engineering Lag and Rolling Window Features ---")
final_df = final_df.sort_values(by=['rm_id', 'date_arrival'])

final_df['lag_7_days'] = final_df.groupby('rm_id')['net_weight'].shift(7)
final_df['lag_14_days'] = final_df.groupby('rm_id')['net_weight'].shift(14)
final_df['lag_28_days'] = final_df.groupby('rm_id')['net_weight'].shift(28)

windows = [7, 14, 28]
for window in windows:
    final_df[f'rolling_mean_{window}_days'] = final_df.groupby('rm_id')['net_weight'].shift(1).rolling(window=window).mean()
    final_df[f'rolling_std_{window}_days'] = final_df.groupby('rm_id')['net_weight'].shift(1).rolling(window=window).std()

final_df.fillna(0, inplace=True)

print("\n--- Engineering State-Based Features ---")
final_df['yearly_cum_net_weight'] = final_df.groupby(['rm_id', 'year'])['net_weight'].cumsum()
final_df['monthly_cum_net_weight'] = final_df.groupby(['rm_id', 'month'])['net_weight'].cumsum()

# Time Since Last Delivery
final_df['last_delivery_date'] = final_df['date_arrival'].where(final_df['net_weight'] > 0)
final_df['last_delivery_date'] = final_df.groupby('rm_id')['last_delivery_date'].ffill()
final_df['time_since_last_delivery'] = (final_df['date_arrival'] - final_df['last_delivery_date']).dt.days

# Correctly fill and cap the feature
final_df['time_since_last_delivery'].fillna(999, inplace=True)
final_df['time_since_last_delivery'] = final_df['time_since_last_delivery'].clip(upper=999)
final_df = final_df.drop(columns=['last_delivery_date'])
print("All features created successfully.")

print("\n--- Finalizing and Saving Dataset ---")
features_to_keep = [
    'rm_id', 'date_arrival', 'net_weight', 'year', 'month', 'day',
    'day_of_week', 'week_of_year', 'is_closure_day',
    'lag_7_days', 'lag_14_days', 'lag_28_days',
    'rolling_mean_7_days', 'rolling_mean_14_days', 'rolling_mean_28_days',
    'rolling_std_7_days', 'rolling_std_14_days', 'rolling_std_28_days',
    'yearly_cum_net_weight', 'monthly_cum_net_weight', 'time_since_last_delivery'
]

model_ready_df = final_df[features_to_keep]

# Save to Feather format for speed and type preservation
output_path1 = 'data/mod_data/model_ready_data.feather'
output_path2 = 'data/mod_data/model_ready_data.csv'
model_ready_df.to_feather(output_path1)
model_ready_df.to_csv(output_path2, index=False)

print("Final dataset created successfully!")
print(model_ready_df.head())
print(f"\nDataset saved to '{output_path1}'")
print(f"\nDataset saved to '{output_path2}'")
