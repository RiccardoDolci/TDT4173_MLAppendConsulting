# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your datasets first
receivals_df = pd.read_csv('data/mod_data/mod_receivals.csv', sep=';')

# Convert the columns using pd.to_datetime handling potential errors
receivals_df['date_arrival'] = pd.to_datetime(receivals_df['date_arrival'], errors='coerce', utc=True)


# Check how many rows have a missing rm_id before cleaning
print(f"Rows before dropping missing rm_id: {len(receivals_df)}")
missing_rm_id_count = receivals_df['rm_id'].isnull().sum()
print(f"Number of rows with missing rm_id: {missing_rm_id_count}")

# Check how many rows have a missing date_arrival before cleaning
print(f"Rows before dropping missing date_arrival: {len(receivals_df)}")
missing_date_arrival_count = receivals_df['date_arrival'].isnull().sum()
print(f"Number of rows with missing date_arrival: {missing_date_arrival_count}")

# Drop rows where 'rm_id' is missing and update the DataFrame
receivals_df.dropna(subset=['rm_id',], inplace=True)
# Add this line to convert the column back to integer
receivals_df['rm_id'] = receivals_df['rm_id'].astype(int)


# Drop rows where 'date_arrival' is missing and update the DataFrame
receivals_df.dropna(subset=['date_arrival',], inplace=True)
# Check how many rows have a missing date_arrival after cleaning
print(f"Rows after dropping missing date_arrival: {len(receivals_df)}")

# Check for negative net_weight values
negative_weight_count = (receivals_df['net_weight'] < 0).sum()
print(f"Found {negative_weight_count} rows with negative net_weight.")
# Replace negative values with NaN 
receivals_df.loc[receivals_df['net_weight'] < 0, 'net_weight'] = np.nan

# Check for zero net_weight values
zero_weight_count = (receivals_df['net_weight'] == 0).sum()
print(f"Found {zero_weight_count} rows with zero net_weight.")

# Calculate IQR for net_weight
Q1 = receivals_df['net_weight'].quantile(0.25)
Q3 = receivals_df['net_weight'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = receivals_df[(receivals_df['net_weight'] < lower_bound) | (receivals_df['net_weight'] > upper_bound)]
print(f"Found {len(outliers)} outliers based on IQR.")

# Define the path for your new, clean file
output_path = 'data/mod_data/clean_receivals.csv'

# Save the DataFrame to a new CSV file
# index=False prevents pandas from writing the DataFrame index as a column
receivals_df.to_csv(output_path, index=False)

print(f"Cleaned data successfully saved to {output_path}")