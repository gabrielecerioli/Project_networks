import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

# ----------------------------------------------------------------------
# 1. Load CEPII Gravity with Mirror-Combined Trade Flows
# ----------------------------------------------------------------------
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
cols_to_use = ['year','iso3_o','iso3_d','tradeflow_comtrade_o','tradeflow_comtrade_d']

print(f"Loading CEPII Gravity (year={TARGET_YEAR}) in chunks…")
t0 = time.time()
chunks = []
for chunk in pd.read_csv(gravity_file, usecols=cols_to_use, chunksize=500_000):
    chunk = chunk[chunk['year'] == TARGET_YEAR]
    if not chunk.empty:
        chunks.append(chunk)
df_trade = pd.concat(chunks, ignore_index=True)
print(f"  → {len(df_trade)} bilateral rows for {TARGET_YEAR} loaded in {time.time() - t0:.2f}s.")

# Combine origin-report and destination-report (mirror statistics)
df_trade['val'] = df_trade['tradeflow_comtrade_o'].fillna(df_trade['tradeflow_comtrade_d'])

orig_only = df_trade['tradeflow_comtrade_o'].notna().sum()
dest_only = df_trade['tradeflow_comtrade_d'].notna().sum()
combined  = df_trade['val'].notna().sum()
print(f"  Origin-report coverage:      {orig_only} / {len(df_trade)}")
print(f"  Destination-report coverage: {dest_only} / {len(df_trade)}")
print(f"  ★ Combined mirror coverage:  {combined} / {len(df_trade)}")

# Filter: positive trade, distinct countries
df_trade = df_trade[
    (df_trade['val'] > 0) &
    (df_trade['iso3_o'] != df_trade['iso3_d'])
].copy()

# Total trade per country (as both exporter and importer)
trade_by_country = (
    df_trade.groupby('iso3_o')['val'].sum()
    .add(df_trade.groupby('iso3_d')['val'].sum(), fill_value=0)
)

# Select top-N nations with valid coordinates
all_countries = set(df_trade['iso3_o']).union(set(df_trade['iso3_d']))
valid = all_countries.intersection(set(iso_to_lat.keys()))
top_series = trade_by_country.loc[list(valid)].nlargest(N_NATIONS)
top_iso = set(top_series.index)
N_NATIONS = len(top_iso)

# FIX: Order by total_trade descending, NOT alphabetically!
df_top = pd.DataFrame({
    "iso":  top_series.index,
    "name": [iso_to_name.get(c, c) for c in top_series.index],
    "lat":  [iso_to_lat[c]  for c in top_series.index],
    "lon":  [iso_to_lon[c]  for c in top_series.index],
    "total_trade": top_series.values
}).reset_index(drop=True)

iso_to_idx = {iso: i for i, iso in enumerate(df_top["iso"])}

# Filter trade to top-N
trade_agg = df_trade[
    df_trade['iso3_o'].isin(top_iso) & df_trade['iso3_d'].isin(top_iso)
].copy()
trade_agg = trade_agg.groupby(['iso3_o','iso3_d'])['val'].sum().reset_index()
trade_agg.rename(columns={'iso3_o':'src','iso3_d':'dst'}, inplace=True)

print(f"\nLoaded {len(trade_agg)} bilateral trade flows among top {N_NATIONS} nations.")
print(f"Top 10 exporters: {', '.join(df_top['name'].head(10).values)}")
print(f"Top 10 ISO:      {', '.join(df_top['iso'].head(10).values)}")
print("Top 10 details:\n", df_top.head(10))

# ----------------------------------------------------------------------
# 2. Build 3D Spherical Coordinates & Weighted Adjacency
# ----------------------------------------------------------------------
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

r0     = float(2.0 * D_phys.min(axis=1).mean())
log_r0 = float(np.log(r0))
print(f"\nMean node strength c = {c_target:.2f} | "
      f"Distance: [{D_phys.min():.0f}-{D_phys[iu,ju].max():.0f}] km | r0 = {r0:.0f} km")

# ----------------------------------------------------------------------
# 3. Gauge-Invariant Profile Likelihood Inference
# ----------------------------------------------------------------------
t_inf = time.time()
SIGMA_GAUGE = 1.0; d_s = 2

D_phys_t = torch.tensor(D_phys, dtype=torch.float32, device=device)
W_obs_u  = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
iu_t     = torch.tensor(iu, dtype=torch.long, device=device)
ju_t     = torch.tensor(ju, dtype=torch.long, device=device)
eye_mask = 1.0 - torch.eye(N_NATIONS, device=device)

# SVD Residual Init
K_sp0 = np.exp(-D_phys / r0)
np.fill_diagonal(K_sp0, 0.0)
s_sp0 = K_sp0.sum(axis=1, keepdims=True) + 1e-12
W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
R_init = np.maximum(0.0, W - W_sp0)

U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
S_init = U[:, :d_s] * np.sqrt(sv[:d_s])
S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0

logit_lam = torch.tensor(0.0, requires_grad=True, device=device)
log_r     = torch.tensor(log_r0, requires_grad=True, device=device)
S_param   = torch.tensor(S_init, requires_grad=True, device=device)

# Adam Warmup (50 steps)
adam_opt = torch.optim.Adam([
    {"params":[logit_lam],"lr":0.08},
    {"params":[log_r],    "lr":0.05},
    {"params":[S_param],  "lr":0.10}
])
for step in range(50):
    adam_opt.zero_grad()
    lam = torch.sigmoid(logit_lam)
    r   = torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3))
    K_sp = torch.exp(-D_phys_t / r) * eye_mask
    diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
    D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(N_NATIONS,device=device)
    K_soc = torch.exp(-D_soc / SIGMA_GAUGE) * eye_mask
    W_raw_t = (1-lam)*K_sp + lam*K_soc
    rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
    W_pred = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
    wu = W_pred[iu_t, ju_t]
    loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
    loss.backward()
    adam_opt.step()

# L-BFGS Polish (45 iter)
lbfgs = torch.optim.LBFGS([logit_lam,log_r,S_param], lr=0.5,
                           max_iter=45, history_size=10,
                           line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad()
    lam = torch.sigmoid(logit_lam)
    r   = torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3))
    K_sp = torch.exp(-D_phys_t / r) * eye_mask
    diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
    D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(N_NATIONS,device=device)
    K_soc = torch.exp(-D_soc / SIGMA_GAUGE) * eye_mask
    W_raw_t = (1-lam)*K_sp + lam*K_soc
    rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
    W_pred = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
    wu = W_pred[iu_t, ju_t]
    loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
    loss.backward()
    return loss
lbfgs.step(closure)

lam_hat = float(torch.sigmoid(logit_lam).item())
r_hat   = float(torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3)).item())
loss_c  = closure().item()

# Fisher Profile 95% CI
dl = 0.04; pl = [loss_c]
for off in [-dl, dl]:
    lt = torch.tensor(float(np.clip(lam_hat+off,0.01,0.99)), device=device)
    lr_p = log_r.clone().detach().requires_grad_(True)
    S_p  = S_param.clone().detach().requires_grad_(True)
    opt_p = torch.optim.LBFGS([lr_p,S_p], lr=0.5, max_iter=25,
                               history_size=10, line_search_fn="strong_wolfe")
    def pc():
        opt_p.zero_grad()
        r = torch.exp(torch.clamp(lr_p, log_r0-3, log_r0+3))
        K_sp = torch.exp(-D_phys_t/r)*eye_mask
        diff = S_p.unsqueeze(1)-S_p.unsqueeze(0)
        D_soc = torch.sqrt((diff**2).sum(-1)+1e-12)+1e9*torch.eye(N_NATIONS,device=device)
        K_soc = torch.exp(-D_soc/SIGMA_GAUGE)*eye_mask
        W_raw_t = (1-lt)*K_sp + lt*K_soc
        rs = W_raw_t.sum(dim=1,keepdim=True)+1e-12
        W_pred = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
        wu = W_pred[iu_t,ju_t]
        loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((lr_p-log_r0)/2)**2
        loss.backward()
        return loss
    opt_p.step(pc); pl.append(pc().item())

d2 = (pl[2]-2*pl[0]+pl[1])/(dl**2)
lam_std = float(np.sqrt(1.0/d2)) if d2 > 1e-3 else 0.03
ci_lo = float(np.clip(lam_hat-1.96*lam_std, 0, 1))
ci_hi = float(np.clip(lam_hat+1.96*lam_std, 0, 1))
runtime = time.time() - t_inf
S_hat = S_param.detach().cpu().numpy()

# ----------------------------------------------------------------------
# 4. Results
# ----------------------------------------------------------------------
print("="*65)
print(f"WORLD COMMERCIAL NETWORK INFERENCE COMPLETE ({runtime:.1f}s)")
print("="*65)
print(f"• λ (Geopolitical weight)  : {lam_hat:.4f} ± {lam_std:.4f}")
print(f"• 95% CI                   : [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"• r (Trade distance scale)  : {r_hat:.1f} km")
print(f"• Geopolitical bloc (λ)    : {lam_hat*100:.1f}%")
print(f"• Gravity / proximity (1-λ): {(1-lam_hat)*100:.1f}%")
print("="*65)
