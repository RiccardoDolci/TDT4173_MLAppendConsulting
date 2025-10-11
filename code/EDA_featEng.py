# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import product
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df = pd.read_csv('data/mod_data/merged_clean_data.csv', sep=';')

df['date_arrival'] = pd.to_datetime(df['date_arrival']).dt.date

rm_ids = df['rm_id'].unique()
dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='D')

# 1. Create the Cartesian product
all_combinations = product(rm_ids, dates)

# 2. Create the master DataFrame from these combinations
master_df = pd.DataFrame(all_combinations, columns=['rm_id', 'date_arrival'])


master_df['date_arrival'] = pd.to_datetime(master_df['date_arrival']).dt.date

# Now, let's perform the left merge
final_df = pd.merge(master_df, df, on=['rm_id', 'date_arrival'], how='left')
# Replace NaN in 'net_weight' with 0
final_df['net_weight'] = final_df['net_weight'].fillna(0)
# Convert the 'date_arrival' column back to datetime objects
final_df['date_arrival'] = pd.to_datetime(final_df['date_arrival'])
# Create year, month, and day features
final_df['year'] = final_df['date_arrival'].dt.year
final_df['month'] = final_df['date_arrival'].dt.month
final_df['day'] = final_df['date_arrival'].dt.day
final_df['day_of_week'] = final_df['date_arrival'].dt.dayofweek
final_df['week_of_year'] = final_df['date_arrival'].dt.isocalendar().week

# Group by rm_id, select the net_weight column, and then calculate the cumulative sum
final_df['daily_cum_net_weight'] = final_df.groupby('rm_id')['net_weight'].cumsum()

# This is crucial for lag/rolling features to work correctly
final_df = final_df.sort_values(by=['rm_id', 'date_arrival'])

# --- Create Lag Features ---
# We group by 'rm_id' to make sure we're not leaking data from one material to another
final_df['lag_7_days'] = final_df.groupby('rm_id')['net_weight'].shift(7)
final_df['lag_14_days'] = final_df.groupby('rm_id')['net_weight'].shift(14)
final_df['lag_28_days'] = final_df.groupby('rm_id')['net_weight'].shift(28)

# --- Create Rolling Window Features for Mean and Standard Deviation ---
# We shift by 1 to ensure we only use data from *before* the current day.

windows = [7, 14, 28]

for window in windows:
    # Rolling Mean
    final_df[f'rolling_mean_{window}_days'] = final_df.groupby('rm_id')['net_weight'].shift(1).rolling(window=window).mean()
    
    # Rolling Standard Deviation
    final_df[f'rolling_std_{window}_days'] = final_df.groupby('rm_id')['net_weight'].shift(1).rolling(window=window).std()


# The rolling features create NaNs for the initial periods. We'll fill them with 0.
final_df.fillna(0, inplace=True)

# --- Create the 'time_since_last_delivery' feature ---

# 1. Create a helper column that only contains the date if there was a delivery
final_df['last_delivery_date'] = final_df['date_arrival'].where(final_df['net_weight'] > 0)

# 2. Forward-fill the last delivery date within each rm_id group
# This fills the NaN values with the date of the last known delivery
final_df['last_delivery_date'] = final_df.groupby('rm_id')['last_delivery_date'].ffill()

# 3. Calculate the difference in days between the current date and the last delivery date
final_df['time_since_last_delivery'] = (final_df['date_arrival'] - final_df['last_delivery_date']).dt.days

# 4. Clean up
# For the period before the very first delivery of an rm_id, the value will be NaN.
# We can fill this with 0 or a large number. Let's start with 0.
final_df['time_since_last_delivery'] = final_df['time_since_last_delivery'].fillna(0)

# Drop the helper column as it's no longer needed
final_df = final_df.drop(columns=['last_delivery_date'])

# --- List of columns to keep for the model ---
features_to_keep = [
    # Identifiers
    'rm_id',
    'date_arrival',

    # Target
    'net_weight',

    # Date Features
    'year',
    'month',
    'day',
    'day_of_week', # Assuming we created this
    'week_of_year', # Assuming we created this

    # Lag Features
    'lag_7_days',
    'lag_14_days',
    'lag_28_days',

    # Rolling Window Features
    'rolling_mean_7_days',
    'rolling_mean_14_days',
    'rolling_mean_28_days',
    'rolling_std_7_days',
    'rolling_std_14_days',
    'rolling_std_28_days',

    # Other Engineered Features
    'daily_cum_net_weight',
    'time_since_last_delivery'
]

# Create the final, clean DataFrame
model_ready_df = final_df[features_to_keep]

# --- Save the dataset to a new file ---
model_ready_df.to_csv('data/mod_data/model_ready_data.csv')

print("Final dataset created successfully!")
print(model_ready_df.head())
print(f"\nDataset saved to 'data/mod_data/model_ready_data.csv'")