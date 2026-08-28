import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

gravity_file = "../data/World_commercial_network/Gravity_V202211.csv"
coords_file  = "../data/World_commercial_network/countries_codes_and_coordinates.csv"
TARGET_YEAR  = 1988
N_NATIONS    = 100

# --- Fallback coordinates ---
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

# --- Read only needed columns in chunks ---
cols_to_use = ['year','iso3_o','iso3_d',
               'tradeflow_comtrade_o','tradeflow_comtrade_d',
               'tradeflow_baci','tradeflow_imf_o','tradeflow_imf_d',
               'lat_o','lon_o','lat_d','lon_d']

print(f"Loading CEPII Gravity (year={TARGET_YEAR}) in chunks…")
t0 = time.time()
chunks = []
for chunk in pd.read_csv(gravity_file, usecols=lambda c: c in cols_to_use,
                          chunksize=500_000):
    chunk = chunk[chunk['year'] == TARGET_YEAR]
    if not chunk.empty:
        chunks.append(chunk)
df_trade = pd.concat(chunks, ignore_index=True)
print(f"  → {len(df_trade)} bilateral rows for {TARGET_YEAR} loaded in {time.time() - t0:.2f}s.")

print("Coverage diagnostics:")
for col in ['tradeflow_comtrade_o', 'tradeflow_comtrade_d', 'tradeflow_baci', 'tradeflow_imf_o', 'tradeflow_imf_d']:
    if col in df_trade.columns:
        print(f"  {col}: {df_trade[col].notna().sum()} / {len(df_trade)}")

# Check combining comtrade_o, comtrade_d, imf_o, imf_d
val_comtrade = df_trade['tradeflow_comtrade_o'].fillna(df_trade['tradeflow_comtrade_d'])
val_all = val_comtrade.fillna(df_trade['tradeflow_imf_o']).fillna(df_trade['tradeflow_imf_d'])
print(f"  Combined comtrade: {val_comtrade.notna().sum()}")
print(f"  Combined all (comtrade+imf): {val_all.notna().sum()}")
