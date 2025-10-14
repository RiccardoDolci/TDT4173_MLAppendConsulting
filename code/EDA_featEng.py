import pandas as pd
import numpy as np

# Load the receivals.csv file
receivals = pd.read_csv('data/mod_data/mod_receivals.csv', sep=';')

# Identify the rm_id column (you might need to adjust this line)
rmid_column = [col for col in receivals.columns if 'rm' in col.lower()][0]

# Identify the date/arrival column (adjust as needed)
date_col = [col for col in receivals.columns if 'date' in col.lower() or 'arrival' in col.lower()][0]
receivals[date_col] = pd.to_datetime(receivals[date_col].astype(str).str[:10], errors='coerce')
weight_col = [col for col in receivals.columns if 'weight' in col.lower()][0]

# Drop rows without a valid date(only date part)
receivals[date_col] = pd.to_datetime(receivals[date_col].astype(str).str[:10], errors='coerce')
receivals = receivals.dropna(subset=[date_col])

# Extract year and day of year
receivals['year'] = receivals[date_col].dt.year
receivals['day_of_year'] = receivals[date_col].dt.dayofyear

# Get all unique rm_ids and years
rm_ids = sorted(receivals[rmid_column].unique())
years = receivals[date_col].dt.year.unique()

# Build the new DataFrame rows: one per day for each year, columns for each rm_id
records = []
for year in years:
    days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    
    # Filter data for the current year only
    df_year = receivals[receivals['year'] == year]
    
    # Prepare a daily cumulative sum dict per rm_id
    cum_sums = {rm: [0]*(days_in_year+1) for rm in rm_ids}  # Index by day_of_year
    
    # Sum weights for each day and rm_id
    for rm in rm_ids:
        daily_sum = df_year[df_year[rmid_column] == rm].groupby('day_of_year')[weight_col].sum()
        # Fill cumulative sums array for each day
        for day in range(1, days_in_year + 1):
            cum_sums[rm][day] = cum_sums[rm][day-1] + daily_sum.get(day, 0)
    
    # Build rows: ID is day from 1 to days_in_year
    for day in range(1, days_in_year + 1):
        row = {'ID': day}
        for rm in rm_ids:
            row[str(rm)] = cum_sums[rm][day]
        records.append(row)

# Create and save the new dataset
new_df = pd.DataFrame(records)
new_df.to_csv('data/mod_data/model_ready_data.csv', index=False)

print(new_df.head())
