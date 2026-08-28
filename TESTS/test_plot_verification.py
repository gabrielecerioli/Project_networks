import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

# ======================================================================
# 1. Load CEPII Gravity with Mirror-Combined Trade Flows
# ======================================================================
gravity_file = "../data/World_commercial_network/Gravity_V202211.csv"
coords_file  = "../data/World_commercial_network/countries_codes_and_coordinates.csv"
TARGET_YEAR  = 1988
N_NATIONS    = 100

# --- Coordinates lookup ---
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

# --- Read needed columns in chunks ---
cols_to_use = ['year','iso3_o','iso3_d','tradeflow_comtrade_o','tradeflow_comtrade_d']

print(f"Loading CEPII Gravity (year={TARGET_YEAR}) in chunks…")
t0 = time.time()
chunks = []
for chunk in pd.read_csv(gravity_file, usecols=cols_to_use, chunksize=500_000):
    chunk = chunk[chunk['year'] == TARGET_YEAR]
    if not chunk.empty:
        chunks.append(chunk)
df_trade = pd.concat(chunks, ignore_index=True)
print(f"  → {len(df_trade)} bilateral rows for {TARGET_YEAR}.")

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

# Total trade per country (exporter + importer)
trade_by_country = (
    df_trade.groupby('iso3_o')['val'].sum()
    .add(df_trade.groupby('iso3_d')['val'].sum(), fill_value=0)
)

# Select top-N nations with valid coordinates, ranked by total trade volume
all_countries = set(df_trade['iso3_o']).union(set(df_trade['iso3_d']))
valid = all_countries.intersection(set(iso_to_lat.keys()))
top_series = trade_by_country.loc[list(valid)].nlargest(N_NATIONS)
top_iso = set(top_series.index)
N_NATIONS = len(top_iso)

# Ranked DataFrame (descending by trade)
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
for _ in range(50):
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

# ----------------------------------------------------------------------
# 5. Visualization
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Panel 1: World Map
ax1 = axes[0,0]
sc1 = ax1.scatter(df_top["lon"], df_top["lat"],
                  c=np.log1p(df_top["total_trade"]), cmap="plasma",
                  s=np.log1p(df_top["total_trade"])*3+15,
                  alpha=0.85, edgecolors="k", lw=0.5)
for i in range(min(12, N_NATIONS)):
    ax1.annotate(df_top.loc[i,"iso"], (df_top.loc[i,"lon"], df_top.loc[i,"lat"]),
                 textcoords="offset points", xytext=(4,4),
                 fontsize=8, fontweight="bold", color="#1e293b")
ax1.set_xlabel("Longitude (deg)"); ax1.set_ylabel("Latitude (deg)")
ax1.set_title("1. Physical Geography of Global Trading Nations", fontweight="bold")
ax1.grid(True, ls="--", alpha=0.4)
plt.colorbar(sc1, ax=ax1).set_label("Log Total Trade", fontsize=9)

# Panel 2: Latent Geopolitical Space
ax2 = axes[0,1]
sc2 = ax2.scatter(S_hat[:,0], S_hat[:,1],
                  c=np.log1p(df_top["total_trade"]), cmap="plasma",
                  s=np.log1p(df_top["total_trade"])*3+15,
                  alpha=0.85, edgecolors="k", lw=0.5)
for i in range(min(15, N_NATIONS)):
    ax2.annotate(df_top.loc[i,"iso"], (S_hat[i,0], S_hat[i,1]),
                 textcoords="offset points", xytext=(4,4),
                 fontsize=9, fontweight="bold", color="#0f172a")
ax2.set_xlabel("Latent Geopolitical Dim 1"); ax2.set_ylabel("Dim 2")
ax2.set_title("2. Inferred Geopolitical Trade Bloc Space (S)", fontweight="bold")
ax2.grid(True, ls="--", alpha=0.4)
plt.colorbar(sc2, ax=ax2).set_label("Log Total Trade", fontsize=9)

# Panel 3: Trade vs Distance
ax3 = axes[1,0]
w_u = W[iu,ju]; d_u = D_phys[iu,ju]; act = w_u > 0
ax3.scatter(d_u[act], w_u[act], color="#6366f1", alpha=0.35, s=14, label="Observed Trade (w_ij)")

# Exact network spatial kernel row-sum normalization
K_sp_hat = np.exp(-D_phys / r_hat)
np.fill_diagonal(K_sp_hat, 0.0)
mean_s_sp = float(K_sp_hat.sum(axis=1).mean())

dc = np.linspace(0, 12700, 200)
dec = (1.0 - lam_hat) * c_target * np.exp(-dc / r_hat) / mean_s_sp
ax3.plot(dc, dec, color="red", lw=2.5, ls="--", label=f"Gravity Decay (r = {r_hat:.0f} km)")
ax3.set_xlabel("Physical Distance (km)"); ax3.set_ylabel("Log Bilateral Trade (w_ij)")
ax3.set_title("3. Bilateral Trade vs Geographical Distance", fontweight="bold")
ax3.set_ylim(bottom=0); ax3.grid(True, ls="--", alpha=0.4); ax3.legend(fontsize=9)

# Panel 4: Lambda
ax4 = axes[1,1]
ax4.errorbar([0],[lam_hat], yerr=[[lam_hat-ci_lo],[ci_hi-lam_hat]],
             fmt="o", color="#d97706", ecolor="#fbbf24", elinewidth=3,
             capsize=8, capthick=2, markersize=10, label=f"λ = {lam_hat:.3f}")
ax4.set_xlim(-0.6,0.6); ax4.set_ylim(0,1); ax4.set_xticks([])
ax4.set_ylabel("Mixing Parameter (λ)")
ax4.set_title(f"4. λ: {lam_hat:.3f} | 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]", fontweight="bold")
ax4.axhline(1.0, color="gray", ls=":", alpha=0.6, label="Pure Bloc (λ=1)")
ax4.axhline(0.5, color="blue", ls="--", alpha=0.5, label="Balanced (λ=0.5)")
ax4.axhline(0.0, color="gray", ls="--", alpha=0.6, label="Pure Distance (λ=0)")
ax4.grid(axis="y", ls="--", alpha=0.5); ax4.legend(loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig("test_output_figure.png", dpi=150)
print("\nPlot saved successfully to test_output_figure.png")
