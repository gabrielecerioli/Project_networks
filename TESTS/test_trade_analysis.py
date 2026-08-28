import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

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
for chunk in pd.read_csv(gravity_file, usecols=cols_to_use, chunksize=500_000):
    chunk = chunk[chunk['year'] == TARGET_YEAR]
    if not chunk.empty:
        chunks.append(chunk)
df_trade = pd.concat(chunks, ignore_index=True)

df_trade['val'] = df_trade['tradeflow_comtrade_o'].fillna(df_trade['tradeflow_comtrade_d'])
df_trade = df_trade[(df_trade['val'] > 0) & (df_trade['iso3_o'] != df_trade['iso3_d'])].copy()

trade_by_country = (
    df_trade.groupby('iso3_o')['val'].sum()
    .add(df_trade.groupby('iso3_d')['val'].sum(), fill_value=0)
)

all_countries = set(df_trade['iso3_o']).union(set(df_trade['iso3_d']))
valid = all_countries.intersection(set(iso_to_lat.keys()))
top_series = trade_by_country.loc[list(valid)].nlargest(N_NATIONS)
top_iso = set(top_series.index)

df_top = pd.DataFrame({
    "iso":  top_series.index,
    "name": [iso_to_name.get(c, c) for c in top_series.index],
    "lat":  [iso_to_lat[c]  for c in top_series.index],
    "lon":  [iso_to_lon[c]  for c in top_series.index],
    "total_trade": top_series.values
}).reset_index(drop=True)

iso_to_idx = {iso: i for i, iso in enumerate(df_top["iso"])}

trade_agg = df_trade[
    df_trade['iso3_o'].isin(top_iso) & df_trade['iso3_d'].isin(top_iso)
].copy()
trade_agg = trade_agg.groupby(['iso3_o','iso3_d'])['val'].sum().reset_index()
trade_agg.rename(columns={'iso3_o':'src','iso3_d':'dst'}, inplace=True)

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

w_u = W[iu, ju]
d_u = D_phys[iu, ju]
act = w_u > 0

pr, pval = pearsonr(d_u[act], w_u[act])
sr, sval = spearmanr(d_u[act], w_u[act])
print(f"Bilateral trade vs distance correlation (active pairs {act.sum()}/{len(w_u)}):")
print(f"  Pearson r = {pr:.4f} (p={pval:.2e})")
print(f"  Spearman rho = {sr:.4f} (p={sval:.2e})")

# Check decay curve values:
# Correct formula for expected spatial link weight at distance d:
# In the network model, for node i, sum_{j} W_{pred, ij} = c_target
# When lambda=0: W_pred_ij = 0.5 * c_target * (K_ij / row_sum_i + K_ji / row_sum_j)
# The average row sum of K_sp is:
r_test = 21232.0
K_sp_test = np.exp(-D_phys / r_test)
np.fill_diagonal(K_sp_test, 0.0)
mean_row_sum = K_sp_test.sum(axis=1).mean()
print(f"Mean row sum of K_sp (r={r_test:.0f}) = {mean_row_sum:.2f}")

dc = np.linspace(0, 12000, 200)
# Correct decay curve:
lam_test = 0.5034
dec_correct = (1 - lam_test) * c_target * np.exp(-dc / r_test) / mean_row_sum
print(f"At d=0: correct dec = {dec_correct[0]:.2f}")
print(f"At d=12000: correct dec = {dec_correct[-1]:.2f}")
print(f"Observed w_u: min={w_u[act].min():.2f}, mean={w_u[act].mean():.2f}, max={w_u[act].max():.2f}")
