# Import required libraries
import pandas as pd
import numpy as np

# Load your datasets first
receivals_df = pd.read_csv('data/mod_data/mod_receivals.csv', sep=';',thousands='.')
orders_df = pd.read_csv('data/mod_data/mod_purchase_orders.csv', sep=';',thousands='.')

# Convert the columns using pd.to_datetime handling potential errors
receivals_df['date_arrival'] = pd.to_datetime(receivals_df['date_arrival'], errors='coerce', utc=True)
orders_df['delivery_date'] = pd.to_datetime(orders_df['delivery_date'], errors='coerce', utc=True)
orders_df['created_date_time'] = pd.to_datetime(orders_df['created_date_time'], errors='coerce', utc=True)
orders_df['modified_date_time'] = pd.to_datetime(orders_df['modified_date_time'], errors='coerce', utc=True)

# Check how many rows have a missing rm_id before cleaning
print(f"Rows before dropping missing rm_id: {len(receivals_df)}")
missing_rm_id_count = receivals_df['rm_id'].isnull().sum()
print(f"Number of rows with missing rm_id: {missing_rm_id_count}")

# Drop rows where 'rm_id' is missing and update the DataFrame
receivals_df.dropna(subset=['rm_id',], inplace=True)
# Add this line to convert the column back to integer
receivals_df['rm_id'] = receivals_df['rm_id'].astype(int)

# Check for negative net_weight values
negative_weight_count = (receivals_df['net_weight'] < 0).sum()
print(f"Found {negative_weight_count} rows with negative net_weight.")

# Check for zero net_weight values, here we drop them but we could also consider other strategies
zero_weight_count = (receivals_df['net_weight'] == 0).sum()
print(f"Found {zero_weight_count} rows with zero net_weight.")
receivals_df.dropna(subset=['net_weight'], inplace=True)

# Drop negative or zero quantities 
orders_df['quantity'] = pd.to_numeric(orders_df['quantity'], errors='coerce')
invalid_quantity_mask = (orders_df['quantity'] <= 0)
print(f"Found {invalid_quantity_mask.sum()} rows with invalid (<= 0) quantity.")

# Drop rows with invalid quantity
orders_df = orders_df[~invalid_quantity_mask]

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

# Now, merge the datasets on 'purchase_order_id' and 'purchase_order_item_no'
merge_keys = ['purchase_order_id', 'purchase_order_item_no']

# normalize types: prefer integers if there are no missing values
receivals_df['purchase_order_id'] = receivals_df['purchase_order_id'].astype('Int64')  # nullable int
receivals_df['purchase_order_item_no'] = receivals_df['purchase_order_item_no'].astype('Int64')

orders_df['purchase_order_id'] = orders_df['purchase_order_id'].astype('Int64')
orders_df['purchase_order_item_no'] = orders_df['purchase_order_item_no'].astype('Int64')
# Perform the merge
merged_df_debug = pd.merge(
    receivals_df,
    orders_df,
    on=merge_keys,
    how='left',     # Use left join to keep all receivals
    indicator=True,  # Add a column to show merge status
    validate='many_to_one'  # Ensure many-to-one merge
)
# Now, check the result of the merge
print("Merge status counts:")
print(merged_df_debug['_merge'].value_counts())
# See the result
print(merged_df_debug.head())
print(f"Shape of receivals_df: {receivals_df.shape}")
print(f"Shape of orders_df: {orders_df.shape}")
print(f"Shape of merged_df: {merged_df_debug.shape}")
unmatched_receivals = merged_df_debug[merged_df_debug['_merge'] == 'left_only']
print(unmatched_receivals.head())
# Define the path for your new, clean file
output_path = 'data/mod_data/merged_clean_data.csv'

# Save the DataFrame to a new CSV file
# index=False prevents pandas from writing the DataFrame index as a column
merged_df_debug.to_csv(output_path, index=False)

print(f"Merged data successfully saved to {output_path}")