import time, torch, numpy as np, pandas as pd

baci_dir = "../data/World_commercial_network/BACI_2017_2024_dataset"
coords_file = "../data/World_commercial_network/countries_codes_and_coordinates.csv"

# Load country codes mapping: numeric code -> ISO3
df_cc = pd.read_csv(f"{baci_dir}/country_codes_V202601.csv")
code_to_iso3 = dict(zip(df_cc["country_code"], df_cc["country_iso3"]))
code_to_name = dict(zip(df_cc["country_code"], df_cc["country_name"]))

print(f"Loaded {len(code_to_iso3)} country codes.")

# Load coordinates
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

# Test BACI for 2024
t0 = time.time()
baci_2024_file = f"{baci_dir}/BACI_HS17_Y2024_V202601.csv"
print(f"Reading and aggregating BACI 2024 ({baci_2024_file})...")

df_baci = pd.read_csv(baci_2024_file, usecols=['i', 'j', 'v'])
print(f"Read {len(df_baci)} product-level rows in {time.time() - t0:.2f}s")

# Aggregate bilateral trade volume
t_agg = time.time()
trade_agg = df_baci.groupby(['i', 'j'])['v'].sum().reset_index()
print(f"Aggregated into {len(trade_agg)} bilateral pairs in {time.time() - t_agg:.2f}s")

# Map to ISO3
trade_agg['iso3_o'] = trade_agg['i'].map(code_to_iso3)
trade_agg['iso3_d'] = trade_agg['j'].map(code_to_iso3)
trade_agg = trade_agg.dropna(subset=['iso3_o', 'iso3_d']).copy()
trade_agg = trade_agg[(trade_agg['v'] > 0) & (trade_agg['iso3_o'] != trade_agg['iso3_d'])]

# Compute total trade per nation
trade_by_country = (
    trade_agg.groupby('iso3_o')['v'].sum()
    .add(trade_agg.groupby('iso3_d')['v'].sum(), fill_value=0)
)

valid = set(trade_by_country.index).intersection(set(iso_to_lat.keys()))
top_10 = trade_by_country.loc[list(valid)].nlargest(10)
print("\nTop 10 Global Trading Nations in 2024:")
for rank, (c, v) in enumerate(top_10.items(), 1):
    print(f"  {rank:2d}. {iso_to_name.get(c, c)} ({c}): {v/1e6:,.1f} Billion USD")
