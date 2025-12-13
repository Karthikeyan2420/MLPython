import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# ---------------------------------------
# 1. LOAD LARGE CSV FILE
# ---------------------------------------
# Replace with your CSV name
csv_path = "dataset.csv"

# Dask loads data lazily without consuming full memory
df = dd.read_csv(csv_path, assume_missing=True)

print("Columns:", df.columns)
print("Sample:")
print(df.head())

# ---------------------------------------
# 2. BASIC CLEANING
# ---------------------------------------
df["event_date"] = dd.to_datetime(df["event_time"]).dt.date
df = df[df["amount"] > 0]  # filter invalid values

# ---------------------------------------
# 3. ANALYTICS
# ---------------------------------------

# Total rows
with ProgressBar():
    total_transactions = df.shape[0].compute()
print("Total transactions:", total_transactions)

# Daily metrics
daily = df.groupby("event_date").agg({
    "transaction_id": "count",
    "amount": "sum",
    "is_purchase": "sum"
}).reset_index()

print("\nDaily metrics:")
with ProgressBar():
    print(daily.compute().tail(10))

# Country-level revenue
country_rev = df.groupby("country").agg({
    "amount": "sum",
    "is_purchase": "sum"
}).reset_index()

print("\nCountry revenue:")
with ProgressBar():
    print(country_rev.compute().sort_values("amount", ascending=False))

# Device conversion rate
device_conv = df.groupby("device").agg({
    "is_purchase": "sum",
    "transaction_id": "count"
}).reset_index()

device_conv["conversion_rate"] = (
    device_conv["is_purchase"] / device_conv["transaction_id"]
)

print("\nDevice conversion rate:")
with ProgressBar():
    print(device_conv.compute())
