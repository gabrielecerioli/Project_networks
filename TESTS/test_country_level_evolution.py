import os, time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

# File paths
gravity_file = "../data/World_commercial_network/Gravity_V202211.csv"
coords_file  = "../data/World_commercial_network/countries_codes_and_coordinates.csv"
baci_dir     = "../data/World_commercial_network/BACI_2017_2024_dataset"

# 1. Coordinates lookup
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

# BACI country code mapping
df_cc = pd.read_csv(f"{baci_dir}/country_codes_V202601.csv")
baci_code_to_iso3 = dict(zip(df_cc["country_code"], df_cc["country_iso3"]))

# Model hyperparameters
N_NATIONS   = 100
R_EARTH     = 6371.0
SIGMA_GAUGE = 1.0
d_s         = 2

# Select representative years spanning 1965 to 2024
years_to_analyze = [1965, 1975, 1985, 1995, 2005, 2015, 2018, 2020, 2022, 2024]
hist_years = [y for y in years_to_analyze if y < 2017]
baci_years = [y for y in years_to_analyze if y >= 2017]

print(f"Historical years: {hist_years}")
print(f"BACI years: {baci_years}")

# Load historical data
t_h = time.time()
cols_to_use = ['year','iso3_o','iso3_d','tradeflow_baci','tradeflow_comtrade_o','tradeflow_comtrade_d','tradeflow_imf_o','tradeflow_imf_d']
chunks = []
for chunk in pd.read_csv(gravity_file, usecols=cols_to_use, chunksize=500_000):
    sub = chunk[chunk['year'].isin(hist_years)]
    if not sub.empty:
        chunks.append(sub)
df_hist = pd.concat(chunks, ignore_index=True)
df_hist['val'] = (
    df_hist['tradeflow_baci']
    .fillna(df_hist['tradeflow_comtrade_o'])
    .fillna(df_hist['tradeflow_comtrade_d'])
    .fillna(df_hist['tradeflow_imf_o'])
    .fillna(df_hist['tradeflow_imf_d'])
)
print(f"Loaded historical data in {time.time() - t_h:.2f}s.")

# Tracking key countries of interest
key_countries = ["USA", "CHN", "DEU", "JPN", "GBR", "FRA", "ITA", "RUS", "IND", "CAN", "BRA", "KOR", "MEX", "AUS", "SAU"]
country_trajectories = {iso: [] for iso in key_countries}
country_trade_shares = {iso: [] for iso in key_countries}
country_gravity_shares = {iso: [] for iso in key_countries}
global_metrics = []

# Baseline S_hat for Procrustes alignment across years
S_ref = None
ref_iso_list = None

for yr in years_to_analyze:
    t_yr = time.time()
    if yr in hist_years:
        df_yr = df_hist[(df_hist['year'] == yr) & (df_hist['val'] > 0) & (df_hist['iso3_o'] != df_hist['iso3_d'])].copy()
        trade_agg = df_yr.groupby(['iso3_o','iso3_d'])['val'].sum().reset_index()
    else:
        baci_file = f"{baci_dir}/BACI_HS17_Y{yr}_V202601.csv"
        df_b = pd.read_csv(baci_file, usecols=['i', 'j', 'v'])
        trade_agg = df_b.groupby(['i', 'j'])['v'].sum().reset_index()
        trade_agg['iso3_o'] = trade_agg['i'].map(baci_code_to_iso3)
        trade_agg['iso3_d'] = trade_agg['j'].map(baci_code_to_iso3)
        trade_agg = trade_agg.dropna(subset=['iso3_o', 'iso3_d']).copy()
        trade_agg = trade_agg[(trade_agg['v'] > 0) & (trade_agg['iso3_o'] != trade_agg['iso3_d'])]
        trade_agg.rename(columns={'v': 'val'}, inplace=True)
    
    trade_by_country = (
        trade_agg.groupby('iso3_o')['val'].sum()
        .add(trade_agg.groupby('iso3_d')['val'].sum(), fill_value=0)
    )
    total_world_trade = trade_by_country.sum() / 2.0
    
    all_c = set(trade_agg['iso3_o']).union(set(trade_agg['iso3_d']))
    valid = all_c.intersection(set(iso_to_lat.keys()))
    top_series = trade_by_country.loc[list(valid)].nlargest(N_NATIONS)
    top_iso = set(top_series.index)
    n_nodes = len(top_iso)
    
    df_top = pd.DataFrame({
        "iso": top_series.index,
        "lat": [iso_to_lat[c] for c in top_series.index],
        "lon": [iso_to_lon[c] for c in top_series.index],
        "trade": top_series.values
    }).reset_index(drop=True)
    iso_to_idx = {iso: i for i, iso in enumerate(df_top["iso"])}
    
    # 3D Coordinates
    lat_rad = np.radians(df_top["lat"].values)
    lon_rad = np.radians(df_top["lon"].values)
    x = R_EARTH * np.cos(lat_rad) * np.cos(lon_rad)
    y = R_EARTH * np.cos(lat_rad) * np.sin(lon_rad)
    z = R_EARTH * np.sin(lat_rad)
    X_3d = np.column_stack([x, y, z])
    
    D_phys = cdist(X_3d, X_3d)
    np.fill_diagonal(D_phys, 1e9)
    iu, ju = np.triu_indices(n_nodes, k=1)
    
    trade_top = trade_agg[trade_agg['iso3_o'].isin(top_iso) & trade_agg['iso3_d'].isin(top_iso)]
    W_dir = np.zeros((n_nodes, n_nodes))
    for _, row in trade_top.iterrows():
        W_dir[iso_to_idx[row["iso3_o"]], iso_to_idx[row["iso3_d"]]] += float(row["val"])
    
    W_raw = 0.5 * (W_dir + W_dir.T)
    W = np.log1p(W_raw)
    c_target = float(W.sum(axis=1).mean())
    active_links = int((W > 0).sum() // 2)
    
    r0 = float(2.0 * D_phys.min(axis=1).mean())
    log_r0 = float(np.log(r0))
    
    D_phys_t = torch.tensor(D_phys, dtype=torch.float32, device=device)
    W_obs_u  = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
    iu_t     = torch.tensor(iu, dtype=torch.long, device=device)
    ju_t     = torch.tensor(ju, dtype=torch.long, device=device)
    eye_mask = 1.0 - torch.eye(n_nodes, device=device)
    
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
    
    adam_opt = torch.optim.Adam([
        {"params": [logit_lam], "lr": 0.08},
        {"params": [log_r],     "lr": 0.05},
        {"params": [S_param],   "lr": 0.10}
    ])
    for _ in range(35):
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
    
    lbfgs = torch.optim.LBFGS([logit_lam, log_r, S_param], lr=0.5,
                              max_iter=30, history_size=10, line_search_fn="strong_wolfe")
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
    
    dl = 0.04; pl = [loss_c]
    for off in [-dl, dl]:
        lt = torch.tensor(float(np.clip(lam_hat+off, 0.01, 0.99)), device=device)
        lr_p = log_r.clone().detach().requires_grad_(True)
        S_p  = S_param.clone().detach().requires_grad_(True)
        opt_p = torch.optim.LBFGS([lr_p, S_p], lr=0.5, max_iter=15, line_search_fn="strong_wolfe")
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
    
    S_curr = S_param.detach().cpu().numpy()
    
    # Track country positions in aligned latent space
    # Procrustes alignment to reference year (2024)
    if yr == 2024:
        S_ref = S_curr.copy()
        ref_iso_list = df_top["iso"].tolist()
    
    global_metrics.append({
        "year": yr,
        "lambda": lam_hat,
        "lambda_std": lam_std,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "gravity_share": 1.0 - lam_hat,
        "r_km": r_hat,
        "active_links": active_links
    })
    
    for c in key_countries:
        if c in iso_to_idx:
            idx = iso_to_idx[c]
            country_trajectories[c].append((yr, S_curr[idx, 0], S_curr[idx, 1]))
            country_trade_shares[c].append((yr, df_top.loc[idx, "trade"] / (2.0 * total_world_trade) * 100))
        else:
            country_trajectories[c].append((yr, np.nan, np.nan))
            country_trade_shares[c].append((yr, 0.0))
            
    print(f"Year {yr:4d}: λ = {lam_hat:.4f} [CI: {ci_lo:.4f}, {ci_hi:.4f}] | r = {r_hat:7.1f} km ({time.time() - t_yr:.1f}s)")

print("\nAll years processed successfully.")
