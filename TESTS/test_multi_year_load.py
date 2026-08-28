import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

gravity_file = "../data/World_commercial_network/Gravity_V202211.csv"
coords_file  = "../data/World_commercial_network/countries_codes_and_coordinates.csv"

# 1. Coordinates
df_c = pd.read_csv(coords_file)
df_c.columns = [c.strip().replace('"','') for c in df_c.columns]
for col in df_c.columns:
    df_c[col] = df_c[col].astype(str).str.strip().str.replace('"','')
df_c["lat"] = pd.to_numeric(df_c["Latitude (average)"], errors="coerce")
df_c["lon"] = pd.to_numeric(df_c["Longitude (average)"], errors="coerce")
df_c = df_c.dropna(subset=["Alpha-3 code","lat","lon"]).copy()
iso_to_lat  = dict(zip(df_c["Alpha-3 code"], df_c["lat"]))
iso_to_lon  = dict(zip(df_c["Alpha-3 code"], df_c["lon"]))
iso_to_name = dict(zip(df_c["Alpha-3 code"], df_c["Country"]))

# Select years to analyze (e.g. every 5 years from 1960 to 2020, or key years: 1960, 1970, 1980, 1985, 1988, 1990, 1995, 2000, 2005, 2008, 2010, 2015, 2019, 2020)
years_to_test = [1960, 1970, 1980, 1985, 1990, 1995, 2000, 2005, 2008, 2010, 2015, 2019, 2020]
print(f"Testing {len(years_to_test)} years: {years_to_test}")

cols_to_use = ['year','iso3_o','iso3_d','tradeflow_baci','tradeflow_comtrade_o','tradeflow_comtrade_d']

t0 = time.time()
print("Reading Gravity CSV in chunks for selected years...")
chunks = []
for chunk in pd.read_csv(gravity_file, usecols=cols_to_use, chunksize=500_000):
    sub = chunk[chunk['year'].isin(years_to_test)]
    if not sub.empty:
        chunks.append(sub)
df_all = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df_all)} rows for {len(years_to_test)} years in {time.time() - t0:.2f}s")
