import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

gravity_file = "../data/World_commercial_network/Gravity_V202211.csv"
coords_file  = "../data/World_commercial_network/countries_codes_and_coordinates.csv"
TARGET_YEAR  = 1988
N_NATIONS    = 100

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

# 2. Gravity Data
cols_to_use = ['year','iso3_o','iso3_d','tradeflow_comtrade_o','tradeflow_comtrade_d']
chunks = []
for chunk in pd.read_csv(gravity_file, usecols=lambda c: c in cols_to_use, chunksize=500_000):
    chunk = chunk[chunk['year'] == TARGET_YEAR]
    if not chunk.empty:
        chunks.append(chunk)
df_trade = pd.concat(chunks, ignore_index=True)

# Combine mirror statistics
df_trade['val'] = df_trade['tradeflow_comtrade_o'].fillna(df_trade['tradeflow_comtrade_d'])
df_trade = df_trade[(df_trade['val'] > 0) & (df_trade['iso3_o'] != df_trade['iso3_d'])].copy()

# Country totals
trade_by_country = (
    df_trade.groupby('iso3_o')['val'].sum()
    .add(df_trade.groupby('iso3_d')['val'].sum(), fill_value=0)
)

all_countries = set(df_trade['iso3_o']).union(set(df_trade['iso3_d']))
valid = all_countries.intersection(set(iso_to_lat.keys()))
top_iso_series = trade_by_country.loc[list(valid)].nlargest(N_NATIONS)
top_iso = set(top_iso_series.index)

# Create sorted by total_trade
df_top = pd.DataFrame({
    "iso":  top_iso_series.index,
    "name": [iso_to_name.get(c, c) for c in top_iso_series.index],
    "lat":  [iso_to_lat[c]  for c in top_iso_series.index],
    "lon":  [iso_to_lon[c]  for c in top_iso_series.index],
    "total_trade": top_iso_series.values
}).reset_index(drop=True)

print("Top 10 Trading Nations in 1988:")
print(df_top[["iso", "name", "total_trade"]].head(10))

iso_to_idx = {iso: i for i, iso in enumerate(df_top["iso"])}

# Bilateral trade
trade_agg = df_trade[
    df_trade['iso3_o'].isin(top_iso) & df_trade['iso3_d'].isin(top_iso)
].copy()
trade_agg = trade_agg.groupby(['iso3_o','iso3_d'])['val'].sum().reset_index()
trade_agg.rename(columns={'iso3_o':'src','iso3_d':'dst'}, inplace=True)

# 3D Coordinates
R_EARTH = 6371.0
lat_rad = np.radians(df_top["lat"].values)
lon_rad = np.radians(df_top["lon"].values)
x_geo = R_EARTH * np.cos(lat_rad) * np.cos(lon_rad)
y_geo = R_EARTH * np.cos(lat_rad) * np.sin(lon_rad)
z_geo = R_EARTH * np.sin(lat_rad)
X_3d  = np.column_stack([x_geo, y_geo, z_geo])

D_phys = cdist(X_3d, X_3d)
np.fill_diagonal(D_phys, 1e9)
iu, ju = np.triu_indices(N_NATIONS, k=1)

W_dir = np.zeros((N_NATIONS, N_NATIONS))
for _, row in trade_agg.iterrows():
    W_dir[iso_to_idx[row["src"]], iso_to_idx[row["dst"]]] += float(row["val"])

W_raw = 0.5 * (W_dir + W_dir.T)
W = np.log1p(W_raw)
c_target = float(W.sum(axis=1).mean())

# Compare r0 calculation:
# What is 2.0 * D_phys.min(axis=1).mean()?
min_dist = D_phys.min(axis=1)
r0_1 = float(2.0 * min_dist.mean())
r0_mean = float(D_phys[iu, ju].mean())
r0_median = float(np.median(D_phys[iu, ju]))
print(f"r0 from 2*min = {r0_1:.1f} km, mean dist = {r0_mean:.1f} km, median dist = {r0_median:.1f} km")
print(f"Dist min = {D_phys.min():.1f} km, max = {D_phys[iu, ju].max():.1f} km")

# Let's test the fitting under different settings
