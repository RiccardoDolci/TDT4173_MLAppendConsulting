# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df = pd.read_csv('data/mod_data/merged_clean_data.csv')
df['date_arrival'] = pd.to_datetime(df['date_arrival'], errors='coerce', utc=True)
df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce', utc=True)

# Create a new feature: delivery_delay (in days)
df['delivery_delay'] = (df['date_arrival'] - df['delivery_date']).dt.days
# Create a new feature: fulfillment_ratio
df['fulfillment_ratio'] = df['net_weight'] / df['quantity']

# Set a style
sns.set_style("whitegrid")

# Histogram of delivery_delay
plt.figure(figsize=(10, 6))
sns.histplot(df['delivery_delay'], bins=80, kde=True)
plt.title('Distribution of Delivery Delays (in Days)')
plt.xlabel('Delay (Negative = Early, Positive = Late)')
plt.ylabel('Frequency')
plt.savefig('delivery_delay_distribution.png')
print("Saved delivery delay distribution plot.")
plt.clf() # Clear the plot for the next one

# Box plot of delay by top 10 suppliers
top_suppliers = df['supplier_id'].value_counts().nlargest(10).index
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[df['supplier_id'].isin(top_suppliers)], x='supplier_id', y='delivery_delay', order=top_suppliers)
plt.title('Delivery Delay by Top 10 Suppliers')
plt.xlabel('Supplier ID')
plt.ylabel('Delay (Days)')
plt.xticks(rotation=45)
plt.savefig('delay_by_supplier.png')
print("Saved delay by supplier plot.")

# Scatter plot of net_weight vs quantity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(2000), x='quantity', y='net_weight', alpha=0.5) # Sample to avoid overplotting
plt.title('Net Weight Received vs. Quantity Ordered')
plt.xlabel('Quantity Ordered (kg)')
plt.ylabel('Net Weight Received (kg)')
plt.savefig('weight_vs_quantity_scatter.png')
print("Saved weight vs. quantity scatter plot.")