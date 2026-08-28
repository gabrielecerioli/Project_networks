import pandas as pd
import numpy as np

gravity_file = "../data/World_commercial_network/Gravity_V202211.csv"

# Check available years and trade flow columns
print("Scanning years in Gravity_V202211.csv...")
years = set()
for chunk in pd.read_csv(gravity_file, usecols=['year'], chunksize=1_000_000):
    years.update(chunk['year'].unique())

sorted_years = sorted(list(years))
print(f"Available years: min={min(sorted_years)}, max={max(sorted_years)}")
print(f"Total years available: {len(sorted_years)}")
print(f"Recent years (last 15): {sorted_years[-15:]}")
