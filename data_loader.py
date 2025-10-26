# data_ingestion/data_loader.py

import os
import pandas as pd

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Create synthetic data
data = {
    "feature1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "feature2": [15, 35, 55, 75, 95, 20, 40, 60, 80, 100],
    "target":   [50, 80, 120, 160, 200, 70, 110, 150, 190, 230]
}
df = pd.DataFrame(data)

# Save to CSV
csv_file = "data/sample.csv"
df.to_csv(csv_file, index=False)
print(f"Sample data saved to {csv_file}")
