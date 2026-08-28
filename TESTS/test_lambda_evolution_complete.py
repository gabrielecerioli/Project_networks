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

# Years covering the evolution from Cold War to modern era
years_to_analyze = [1965, 1970, 1975, 1980, 1985, 1988, 1991, 1995, 2000, 2004, 2008, 2012, 2016, 2019, 2020]
N_NATIONS = 100
R_EARTH = 6371.0
SIGMA_GAUGE = 1.0
d_s = 2

cols_to_use = ['year','iso3_o','iso3_d','tradeflow_baci','tradeflow_comtrade_o','tradeflow_comtrade_d','tradeflow_imf_o','tradeflow_imf_d']

print(f"Loading CEPII Gravity for {len(years_to_analyze)} historical years: {years_to_analyze}…")
t_start = time.time()
chunks = []
for chunk in pd.read_csv(gravity_file, usecols=lambda c: c in cols_to_use, chunksize=500_000):
    sub = chunk[chunk['year'].isin(years_to_analyze)]
    if not sub.empty:
        chunks.append(sub)
df_raw = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df_raw)} total rows across {len(years_to_analyze)} years in {time.time() - t_start:.2f}s.")

# Universal tradeflow fallback
df_raw['val'] = (
    df_raw['tradeflow_baci']
    .fillna(df_raw['tradeflow_comtrade_o'])
    .fillna(df_raw['tradeflow_comtrade_d'])
    .fillna(df_raw['tradeflow_imf_o'])
    .fillna(df_raw['tradeflow_imf_d'])
)

results = []

for yr in years_to_analyze:
    t_yr = time.time()
    df_yr = df_raw[(df_raw['year'] == yr) & (df_raw['val'] > 0) & (df_raw['iso3_o'] != df_raw['iso3_d'])].copy()
    
    # Country totals
    trade_by_country = (
        df_yr.groupby('iso3_o')['val'].sum()
        .add(df_yr.groupby('iso3_d')['val'].sum(), fill_value=0)
    )
    all_c = set(df_yr['iso3_o']).union(set(df_yr['iso3_d']))
    valid = all_c.intersection(set(iso_to_lat.keys()))
    top_series = trade_by_country.loc[list(valid)].nlargest(N_NATIONS)
    top_iso = set(top_series.index)
    n_nodes = len(top_iso)
    
    df_top = pd.DataFrame({
        "iso": top_series.index,
        "lat": [iso_to_lat[c] for c in top_series.index],
        "lon": [iso_to_lon[c] for c in top_series.index],
    }).reset_index(drop=True)
    iso_to_idx = {iso: i for i, iso in enumerate(df_top["iso"])}
    
    # 3D Coordinates on Earth sphere
    lat_rad = np.radians(df_top["lat"].values)
    lon_rad = np.radians(df_top["lon"].values)
    x = R_EARTH * np.cos(lat_rad) * np.cos(lon_rad)
    y = R_EARTH * np.cos(lat_rad) * np.sin(lon_rad)
    z = R_EARTH * np.sin(lat_rad)
    X_3d = np.column_stack([x, y, z])
    
    D_phys = cdist(X_3d, X_3d)
    np.fill_diagonal(D_phys, 1e9)
    iu, ju = np.triu_indices(n_nodes, k=1)
    
    # Bilateral trade matrix
    trade_agg = df_yr[df_yr['iso3_o'].isin(top_iso) & df_yr['iso3_d'].isin(top_iso)]
    trade_agg = trade_agg.groupby(['iso3_o','iso3_d'])['val'].sum().reset_index()
    
    W_dir = np.zeros((n_nodes, n_nodes))
    for _, row in trade_agg.iterrows():
        W_dir[iso_to_idx[row["iso3_o"]], iso_to_idx[row["iso3_d"]]] += float(row["val"])
    
    W_raw = 0.5 * (W_dir + W_dir.T)
    W = np.log1p(W_raw)
    c_target = float(W.sum(axis=1).mean())
    active_links = int((W > 0).sum() // 2)
    
    r0 = float(2.0 * D_phys.min(axis=1).mean())
    log_r0 = float(np.log(r0))
    
    # PyTorch Tensors
    D_phys_t = torch.tensor(D_phys, dtype=torch.float32, device=device)
    W_obs_u  = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
    iu_t     = torch.tensor(iu, dtype=torch.long, device=device)
    ju_t     = torch.tensor(ju, dtype=torch.long, device=device)
    eye_mask = 1.0 - torch.eye(n_nodes, device=device)
    
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
    
    # Adam Warmup
    adam_opt = torch.optim.Adam([
        {"params": [logit_lam], "lr": 0.08},
        {"params": [log_r],     "lr": 0.05},
        {"params": [S_param],   "lr": 0.10}
    ])
    for _ in range(40):
        adam_opt.zero_grad()
        lam = torch.sigmoid(logit_lam)
        r   = torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3))
        K_sp = torch.exp(-D_phys_t / r) * eye_mask
        diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
        D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(n_nodes, device=device)
        K_soc = torch.exp(-D_soc / SIGMA_GAUGE) * eye_mask
        W_raw_t = (1-lam)*K_sp + lam*K_soc
        rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
        W_pred = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
        wu = W_pred[iu_t, ju_t]
        loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
        loss.backward(); adam_opt.step()
    
    # L-BFGS Polish
    lbfgs = torch.optim.LBFGS([logit_lam, log_r, S_param], lr=0.5,
                              max_iter=35, history_size=10, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad()
        lam = torch.sigmoid(logit_lam)
        r   = torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3))
        K_sp = torch.exp(-D_phys_t / r) * eye_mask
        diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
        D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(n_nodes, device=device)
        K_soc = torch.exp(-D_soc / SIGMA_GAUGE) * eye_mask
        W_raw_t = (1-lam)*K_sp + lam*K_soc
        rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
        W_pred = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
        wu = W_pred[iu_t, ju_t]
        loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
        loss.backward(); return loss
    lbfgs.step(closure)
    
    lam_hat = float(torch.sigmoid(logit_lam).item())
    r_hat   = float(torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3)).item())
    loss_c  = closure().item()
    
    # Fisher Profile 95% CI
    dl = 0.04; pl = [loss_c]
    for off in [-dl, dl]:
        lt = torch.tensor(float(np.clip(lam_hat+off, 0.01, 0.99)), device=device)
        lr_p = log_r.clone().detach().requires_grad_(True)
        S_p  = S_param.clone().detach().requires_grad_(True)
        opt_p = torch.optim.LBFGS([lr_p, S_p], lr=0.5, max_iter=20, line_search_fn="strong_wolfe")
        def pc():
            opt_p.zero_grad()
            r = torch.exp(torch.clamp(lr_p, log_r0-3, log_r0+3))
            K_sp = torch.exp(-D_phys_t/r)*eye_mask
            diff = S_p.unsqueeze(1)-S_p.unsqueeze(0)
            D_soc = torch.sqrt((diff**2).sum(-1)+1e-12)+1e9*torch.eye(n_nodes, device=device)
            K_soc = torch.exp(-D_soc/SIGMA_GAUGE)*eye_mask
            W_raw_t = (1-lt)*K_sp + lt*K_soc
            rs = W_raw_t.sum(dim=1, keepdim=True)+1e-12
            W_pred = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
            wu = W_pred[iu_t, ju_t]
            loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((lr_p-log_r0)/2)**2
            loss.backward(); return loss
        opt_p.step(pc); pl.append(pc().item())
    
    d2 = (pl[2]-2*pl[0]+pl[1])/(dl**2)
    lam_std = float(np.sqrt(1.0/d2)) if d2 > 1e-3 else 0.03
    ci_lo = float(np.clip(lam_hat-1.96*lam_std, 0, 1))
    ci_hi = float(np.clip(lam_hat+1.96*lam_std, 0, 1))
    
    # Top 3 nations
    top_3_names = [f"{iso_to_name.get(c, c)} ({c})" for c in list(top_series.index[:3])]
    
    results.append({
        "year": yr,
        "lambda": lam_hat,
        "lambda_std": lam_std,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "gravity_share": 1.0 - lam_hat,
        "r_km": r_hat,
        "c_target": c_target,
        "active_links": active_links,
        "top_3": ", ".join(top_3_names),
        "runtime_s": time.time() - t_yr
    })
    print(f"  → Year {yr:4d}: λ = {lam_hat:.4f} ± {lam_std:.4f} [95% CI: {ci_lo:.4f}, {ci_hi:.4f}] | r = {r_hat:7.1f} km | Links = {active_links} ({time.time() - t_yr:.2f}s)")

df_res = pd.DataFrame(results)
print("\n" + "="*85)
print("HISTORICAL EVOLUTION OF GEOPOLITICAL WEIGHT (λ) ACROSS DECADES:")
print("="*85)
print(df_res[["year", "lambda", "ci_lo", "ci_hi", "gravity_share", "r_km", "active_links", "runtime_s"]].to_string(index=False))

# ----------------------------------------------------------------------
# Plot Historical Evolution: Lambda vs Time (Publication Quality)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Panel 1: Lambda vs Year (with 95% Confidence Band)
ax1 = axes[0, 0]
years = df_res["year"].values
lams = df_res["lambda"].values
ci_l = df_res["ci_lo"].values
ci_h = df_res["ci_hi"].values

ax1.fill_between(years, ci_l, ci_h, color="#f59e0b", alpha=0.25, label="95% Profile CI")
ax1.plot(years, lams, marker="o", color="#d97706", lw=2.5, markersize=7, label="Geopolitical Weight (λ)")
ax1.axhline(0.5, color="blue", ls="--", alpha=0.5, label="Balanced Gravity-Bloc (λ = 0.5)")

# Historical Event Annotations
ax1.axvline(1989, color="gray", ls=":", alpha=0.7)
ax1.text(1989.5, 0.58, "Fall of Berlin Wall\n(1989)", fontsize=8, color="#334155", fontweight="bold")

ax1.axvline(2001, color="gray", ls=":", alpha=0.7)
ax1.text(2001.5, 0.32, "China WTO Entry\n(2001)", fontsize=8, color="#334155", fontweight="bold")

ax1.axvline(2008, color="gray", ls=":", alpha=0.7)
ax1.text(2008.5, 0.48, "Global Financial\nCrisis (2008)", fontsize=8, color="#334155", fontweight="bold")

ax1.axvline(2020, color="gray", ls=":", alpha=0.7)
ax1.text(2017.0, 0.52, "COVID-19\nShock (2020)", fontsize=8, color="#dc2626", fontweight="bold")

ax1.set_xlabel("Year", fontsize=10)
ax1.set_ylabel("Mixing Parameter (λ)", fontsize=10)
ax1.set_title("1. Evolution of Geopolitical Bloc Weight λ (1965–2020)", fontweight="bold", fontsize=11)
ax1.set_ylim(0.2, 0.65)
ax1.grid(True, ls="--", alpha=0.4)
ax1.legend(loc="upper right", fontsize=9)

# Panel 2: Gravity vs Geopolitical Share
ax2 = axes[0, 1]
ax2.stackplot(years, df_res["gravity_share"]*100, df_res["lambda"]*100,
              labels=["Gravity / Physical Proximity (1 - λ)", "Geopolitical Alliance Blocs (λ)"],
              colors=["#3b82f6", "#f59e0b"], alpha=0.8)
ax2.set_xlabel("Year", fontsize=10)
ax2.set_ylabel("Fraction of Commercial Network (%)", fontsize=10)
ax2.set_title("2. Trade Decomposition: Gravity vs Geopolitics Share", fontweight="bold", fontsize=11)
ax2.set_ylim(0, 100)
ax2.grid(True, ls="--", alpha=0.4)
ax2.legend(loc="lower left", fontsize=9)

# Panel 3: Network Density & Active Links
ax3 = axes[1, 0]
ax3.plot(years, df_res["active_links"], marker="s", color="#8b5cf6", lw=2.2, markersize=6, label="Active Bilateral Links (w_ij > 0)")
ax3.set_xlabel("Year", fontsize=10)
ax3.set_ylabel("Number of Active Bilateral Links", fontsize=10)
ax3.set_title("3. Growth of Global Trade Connectivity among Top 100 Nations", fontweight="bold", fontsize=11)
ax3.grid(True, ls="--", alpha=0.4)
ax3.legend(loc="upper left", fontsize=9)

# Panel 4: Spatial Distance Scale (r)
ax4 = axes[1, 1]
ax4.plot(years, df_res["r_km"] / 1000.0, marker="^", color="#059669", lw=2.2, markersize=6, label="Spatial Scale r (in '000 km)")
ax4.set_xlabel("Year", fontsize=10)
ax4.set_ylabel("Interaction Distance Scale r (10³ km)", fontsize=10)
ax4.set_title("4. Effective Geographical Trade Reach (r)", fontweight="bold", fontsize=11)
ax4.grid(True, ls="--", alpha=0.4)
ax4.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("test_lambda_evolution_figure.png", dpi=150)
print("\nPlot saved successfully to test_lambda_evolution_figure.png")
