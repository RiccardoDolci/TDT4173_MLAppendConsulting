import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file path (assuming it's in a data/kernel/ path structure)
FILE_PATH = 'data/kernel/purchase_orders.csv'
PLOT_PATH = 'graph/purchase_order_lead_time_distribution.png'

# --- 1. Load Data and Convert Dates ---
print("Loading purchase orders data...")
try:
    # Assuming semicolon delimiter based on data snippet
    orders_df = pd.read_csv(FILE_PATH, sep=';')
except FileNotFoundError:
    print(f"Error: {FILE_PATH} not found. Please check file path.")
    exit()

# Convert required date columns to datetime objects
orders_df['delivery_date'] = pd.to_datetime(orders_df['delivery_date'], errors='coerce', utc=True)
orders_df['created_date_time'] = pd.to_datetime(orders_df['created_date_time'], errors='coerce', utc=True)


# --- 2. Calculate Lead Time ---
# Lead Time = Expected Delivery Date - Creation Date
# This tells us how many days were planned between ordering and expecting delivery.
orders_df['lead_time'] = orders_df['delivery_date'] - orders_df['created_date_time']

# Convert the timedelta object to total days
orders_df['lead_time_days'] = orders_df['lead_time'].dt.days

# Filter out orders where the delivery date was before the creation date (data errors)
orders_df = orders_df[orders_df['lead_time_days'] >= 0].copy()

# Filter out extreme outliers for better visualization (e.g., beyond 5 years)
orders_df = orders_df[orders_df['lead_time_days'] <= 1825].copy()


# --- 3. Visualize the Distribution ---
print("Generating lead time distribution histogram...")
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Use a histogram to show the frequency of different lead times
sns.histplot(
    orders_df['lead_time_days'],
    bins=50,
    kde=True,
    color='#1f77b4',
    edgecolor='black'
)

plt.title('Distribution of Purchase Order Lead Time (Days)', fontsize=14)
plt.xlabel('Lead Time (Days from Order Placement to Expected Delivery)', fontsize=12)
plt.ylabel('Number of Purchase Orders', fontsize=12)
plt.xlim(0, orders_df['lead_time_days'].quantile(0.99)) # Cut off extreme tail
plt.axvline(orders_df['lead_time_days'].median(), color='red', linestyle='--', label=f'Median: {orders_df["lead_time_days"].median():.0f} days')
plt.legend()
plt.tight_layout()

print("\n-- Statistical Parameters for Order Lead Time (Days) --")
print(orders_df['lead_time_days'].describe().to_string())
print("---------------------------------------------------------")
# Save the plot to the specified path
plt.savefig(PLOT_PATH)
print(f"Graph saved to {PLOT_PATH}")