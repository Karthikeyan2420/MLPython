import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --------------------------
# CONFIGURATION
# --------------------------
NUM_ROWS = 10000          # Adjust row count
OUTPUT_CSV = "dataset.csv"
SEED = 42


# --------------------------
# DATA GENERATION
# --------------------------
rng = np.random.default_rng(SEED)

# user IDs
user_ids = rng.integers(1, 2000, size=NUM_ROWS)

# timestamps last 30 days
end = datetime.utcnow()
start = end - timedelta(days=30)
timestamps = rng.integers(int(start.timestamp()), int(end.timestamp()), size=NUM_ROWS)
event_time = pd.to_datetime(timestamps, unit='s')

# categorical columns
countries = ['US', 'GB', 'IN', 'DE', 'CA']
country = rng.choice(countries, size=NUM_ROWS)

devices = ['mobile', 'desktop', 'tablet']
device = rng.choice(devices, size=NUM_ROWS, p=[0.6, 0.3, 0.1])

# numeric fields
amount = np.round(rng.lognormal(mean=2, sigma=0.6, size=NUM_ROWS), 2)
is_purchase = rng.random(NUM_ROWS) < 0.1  # 10% purchase probability

# --------------------------
# BUILD DATAFRAME
# --------------------------
df = pd.DataFrame({
    "transaction_id": range(1, NUM_ROWS + 1),
    "user_id": user_ids,
    "event_time": event_time,
    "country": country,
    "device": device,
    "amount": amount,
    "is_purchase": is_purchase.astype(int)
})

# --------------------------
# SAVE TO CSV
# --------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"CSV file saved as {OUTPUT_CSV}")
