import time, torch, numpy as np, pandas as pd
from scipy.spatial.distance import cdist

gravity_file = "../data/World_commercial_network/Gravity_V202211.csv"
coords_file  = "../data/World_commercial_network/countries_codes_and_coordinates.csv"

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

# Check 2019, 2020, 2021
target_years = [2019, 2020, 2021]
cols_to_use = ['year','iso3_o','iso3_d','tradeflow_baci','tradeflow_comtrade_o','tradeflow_comtrade_d']

for yr in target_years:
    print(f"\n{'='*50}\nScanning year {yr}...")
    t0 = time.time()
    chunks = []
    for chunk in pd.read_csv(gravity_file, usecols=cols_to_use, chunksize=500_000):
        chunk = chunk[chunk['year'] == yr]
        if not chunk.empty:
            chunks.append(chunk)
    df_yr = pd.concat(chunks, ignore_index=True)
    print(f"Total rows for {yr}: {len(df_yr)} (loaded in {time.time() - t0:.2f}s)")
    
    baci_cov = df_yr['tradeflow_baci'].notna().sum()
    com_o_cov = df_yr['tradeflow_comtrade_o'].notna().sum()
    com_d_cov = df_yr['tradeflow_comtrade_d'].notna().sum()
    print(f"  tradeflow_baci: {baci_cov} / {len(df_yr)}")
    print(f"  tradeflow_comtrade_o: {com_o_cov} / {len(df_yr)}")
    print(f"  tradeflow_comtrade_d: {com_d_cov} / {len(df_yr)}")
    
    # Combined trade flow
    df_yr['val'] = df_yr['tradeflow_baci'].fillna(df_yr['tradeflow_comtrade_o']).fillna(df_yr['tradeflow_comtrade_d'])
    active = df_yr[(df_yr['val'] > 0) & (df_yr['iso3_o'] != df_yr['iso3_d'])].copy()
    print(f"  Active flows with positive trade: {len(active)}")
    
    # Country totals
    tot = (active.groupby('iso3_o')['val'].sum()
           .add(active.groupby('iso3_d')['val'].sum(), fill_value=0))
    valid = set(active['iso3_o']).union(set(active['iso3_d'])).intersection(set(iso_to_lat.keys()))
    top_10 = tot.loc[list(valid)].nlargest(10)
    print(f"  Top 10 Trading Nations in {yr}:")
    for rank, (c, v) in enumerate(top_10.items(), 1):
        print(f"    {rank}. {iso_to_name.get(c, c)} ({c}): {v/1e6:.1f} M USD (or thousands)")
