# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Set plot style for visual clarity
plt.style.use('seaborn-v0_8')

# Load your datasets first
receivals_df = pd.read_csv('data/mod_data/mod_receivals.csv', sep=';')
orders_df = pd.read_csv('data/mod_data/mod_purchase_orders.csv', sep=';')

# Convert the columns using pd.to_datetime handling potential errors
receivals_df['date_arrival'] = pd.to_datetime(receivals_df['date_arrival'], errors='coerce', utc=True) # [cite: 57]
orders_df['delivery_date'] = pd.to_datetime(orders_df['delivery_date'], errors='coerce', utc=True) # [cite: 75]
orders_df['created_date_time'] = pd.to_datetime(orders_df['created_date_time'], errors='coerce', utc=True) # [cite: 81]
orders_df['modified_date_time'] = pd.to_datetime(orders_df['modified_date_time'], errors='coerce', utc=True) # [cite: 83]

# Check how many rows have a missing rm_id before cleaning
print(f"Rows before dropping missing rm_id: {len(receivals_df)}")
missing_rm_id_count = receivals_df['rm_id'].isnull().sum()
print(f"Number of rows with missing rm_id: {missing_rm_id_count}")

# Drop rows where 'rm_id' is missing and update the DataFrame
receivals_df.dropna(subset=['rm_id',], inplace=True)
# Add this line to convert the column back to integer
receivals_df['rm_id'] = receivals_df['rm_id'].astype(int)
# Check how many rows have a missing rm_id after cleaning
print(f"Rows after dropping missing rm_id: {len(receivals_df)}")

# Check for negative net_weight values
negative_weight_count = (receivals_df['net_weight'] < 0).sum()
print(f"Found {negative_weight_count} rows with negative net_weight.")

# Replace negative values with NaN so MICE can impute them
receivals_df.loc[receivals_df['net_weight'] < 0, 'net_weight'] = np.nan
# Find and replace negative or zero quantities with NaN
# Any value that cannot be converted will become NaN
orders_df['quantity'] = pd.to_numeric(orders_df['quantity'], errors='coerce')
invalid_quantity_mask = (orders_df['quantity'] <= 0)
print(f"Found {invalid_quantity_mask.sum()} rows with invalid (<= 0) quantity.")

orders_df.loc[invalid_quantity_mask, 'quantity'] = np.nan

# Check for zero net_weight values
zero_weight_count = (receivals_df['net_weight'] == 0).sum()
print(f"Found {zero_weight_count} rows with zero net_weight.")

# DECISION: Zeros are ambiguous. They could be errors or valid.
# For this project, it's safer to treat them as missing data to be imputed.
receivals_df.loc[receivals_df['net_weight'] == 0, 'net_weight'] = np.nan

# Define the conversion factor for LBS to KG
LBS_TO_KG = 0.453592

# Create a mask to identify rows with pounds (unit_id 43)
pounds_mask = (orders_df['unit_id'] == 43)
print(f"Found {pounds_mask.sum()} rows with pounds to convert.")

# Apply the conversion to the 'quantity' column for these rows
orders_df.loc[pounds_mask, 'quantity'] = orders_df.loc[pounds_mask, 'quantity'] * LBS_TO_KG

# Update the unit and unit_id columns for consistency ---

kg_unit_id = 40
orders_df.loc[pounds_mask, 'unit'] = 'kg'
orders_df.loc[pounds_mask, 'unit_id'] = kg_unit_id

print("Pounds have been successfully converted to KG.")

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

# Remove the outliers
receivals_df = receivals_df[(receivals_df['net_weight'] >= lower_bound) & (receivals_df['net_weight'] <= upper_bound)]
print(f"Shape after removing outliers: {receivals_df.shape}")

# Apply a log1p transformation to the quantity column
# log1p is log(1 + x), which safely handles any zero values
orders_df['quantity_log'] = np.log1p(orders_df['quantity'])

# --- PREPARE FOR MICE ---
# IterativeImputer works best with numerical data. Let's select relevant columns.
# We assume 'rm_id', 'product_id', etc. can help predict 'net_weight'.
# Let's ensure these IDs are treated as numbers for the imputer.
cols_for_imputation = ['rm_id', 'product_id', 'purchase_order_id', 'purchase_order_item_no', 'net_weight']

# Create a copy of the subset of the DataFrame to avoid warnings
data_subset = receivals_df[cols_for_imputation].copy()


# --- RUN THE MICE ALGORITHM ---

# 1. Initialize the Imputer
# It will cycle through each column, using the others to predict missing values for 10 rounds (max_iter=10)
mice_imputer = IterativeImputer(max_iter=10, random_state=0)

# 2. Fit and transform the data
# This returns a NumPy array with the missing values filled in
imputed_data = mice_imputer.fit_transform(data_subset)

# 3. Put the imputed data back into the DataFrame
# The result is a NumPy array, so we need to convert it back to a DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=cols_for_imputation)
# Set the index of the imputed data to match the original's index
imputed_df.index = data_subset.index

# --- UPDATE YOUR ORIGINAL DATAFRAME ---

# Now, update the 'net_weight' column in your original DataFrame with the newly imputed values
receivals_df['net_weight'] = imputed_df['net_weight']

# Ensure integer columns are of integer type (algorithm works with floats)
for col in cols_for_imputation:
    receivals_df[col] = receivals_df[col].astype('Int64')  # Use 'Int64' to allow for NaNs if any

# Verify that there are no more missing values in net_weight
print("Missing net_weight values after MICE imputation:")
print(receivals_df['net_weight'].isnull().sum())

# Select relevant numerical columns from orders_df to help the imputer
cols_for_imputation = ['product_id', 'purchase_order_id', 'purchase_order_item_no', 'quantity']
data_subset = orders_df[cols_for_imputation].copy()

# Initialize and run the imputer
mice_imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data = mice_imputer.fit_transform(data_subset)

# Convert the result back to a DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=cols_for_imputation)

# Align the index before assigning the data back
imputed_df.index = data_subset.index

# Update the original DataFrame with the imputed values
orders_df['quantity'] = imputed_df['quantity']

# Ensure integer columns are of integer type (algorithm works with floats)
for col in cols_for_imputation:
    orders_df[col] = orders_df[col].round(0).astype('Int64')  # Use 'Int64' to allow for NaNs if any

# Verify that all missing quantities have been filled
print(f"Missing quantity values after MICE: {orders_df['quantity'].isnull().sum()}")

# Define the keys to merge on
merge_keys = ['purchase_order_id', 'purchase_order_item_no']

# Perform the merge
merged_df_debug = pd.merge(
    receivals_df,
    orders_df,
    on=merge_keys,
    how='left',
    indicator=True    # Add a column to show merge status
)
# Now, check the result of the merge
print("Merge status counts:")
print(merged_df_debug['_merge'].value_counts())
# See the result
print(merged_df_debug.head())
print(f"Shape of receivals_df: {receivals_df.shape}")
print(f"Shape of orders_df: {orders_df.shape}")
print(f"Shape of merged_df: {merged_df_debug.shape}")
# Define the path for your new, clean file
output_path = 'data/mod_data/merged_clean_data.csv'

# Save the DataFrame to a new CSV file
# index=False prevents pandas from writing the DataFrame index as a column
merged_df_debug.to_csv(output_path, index=False)

print(f"Merged data successfully saved to {output_path}")