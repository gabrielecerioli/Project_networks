import os, time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

# ----------------------------------------------------------------------
# Paths setup
# ----------------------------------------------------------------------
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

df_cc = pd.read_csv(f"{baci_dir}/country_codes_V202601.csv")
baci_code_to_iso3 = dict(zip(df_cc["country_code"], df_cc["country_iso3"]))

# Model hyperparameters
N_NATIONS   = 100
R_EARTH     = 6371.0
SIGMA_GAUGE = 1.0
d_s         = 2

# Historical years from Gravity + Recent years from BACI
hist_years = [1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015]
baci_years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
all_years  = sorted(hist_years + baci_years)

# Load historical data in chunks
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

# Countries to track in detail over time
tracked_countries = ["USA", "CHN", "DEU", "JPN", "GBR", "FRA", "ITA", "KOR", "IND", "MEX"]
country_trade_evolution = {iso: [] for iso in tracked_countries}
yearly_results = []

# Latest year data containers
latest_df_top = None
latest_S_hat = None
latest_lam_hat = None
latest_r_hat = None

for yr in all_years:
    if yr in hist_years:
        source_name = "CEPII Gravity"
        df_yr = df_hist[(df_hist['year'] == yr) & (df_hist['val'] > 0) & (df_hist['iso3_o'] != df_hist['iso3_d'])].copy()
        trade_agg = df_yr.groupby(['iso3_o','iso3_d'])['val'].sum().reset_index()
    else:
        source_name = "BACI HS17"
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
        "name": [iso_to_name.get(c, c) for c in top_series.index],
        "lat": [iso_to_lat[c] for c in top_series.index],
        "lon": [iso_to_lon[c] for c in top_series.index],
        "trade": top_series.values
    }).reset_index(drop=True)
    iso_to_idx = {iso: i for i, iso in enumerate(df_top["iso"])}
    
    for c in tracked_countries:
        if c in trade_by_country:
            country_trade_evolution[c].append(float(trade_by_country[c] / (2.0 * total_world_trade) * 100))
        else:
            country_trade_evolution[c].append(0.0)
            
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
    
    yearly_results.append({
        "year": yr,
        "source": source_name,
        "lambda": lam_hat,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "gravity_share": 1.0 - lam_hat,
        "r_km": r_hat,
        "active_links": active_links
    })
    
    if yr == 2024:
        latest_df_top = df_top.copy()
        latest_S_hat = S_param.detach().cpu().numpy()
        latest_lam_hat = lam_hat
        latest_r_hat = r_hat
        
        # Calculate country-level geopolitical vs gravity trade share
        r_t = torch.tensor(r_hat, dtype=torch.float32)
        K_sp_t = torch.exp(-D_phys_t / r_t) * eye_mask
        diff_t = S_param.unsqueeze(1) - S_param.unsqueeze(0)
        D_soc_t = torch.sqrt((diff_t**2).sum(-1)+1e-12) + 1e9*torch.eye(n_nodes, device=device)
        K_soc_t = torch.exp(-D_soc_t / SIGMA_GAUGE) * eye_mask
        
        geo_flux_per_node = (lam_hat * K_soc_t).sum(dim=1).detach().cpu().numpy()
        grav_flux_per_node = ((1.0 - lam_hat) * K_sp_t).sum(dim=1).detach().cpu().numpy()
        latest_df_top["geo_weight"] = geo_flux_per_node / (geo_flux_per_node + grav_flux_per_node + 1e-12) * 100
        latest_df_top["grav_weight"] = 100.0 - latest_df_top["geo_weight"]

df_res = pd.DataFrame(yearly_results)

# ----------------------------------------------------------------------
# 4-Panel Clean, Uncluttered Country-by-Country Dashboard
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(17, 11))
years = df_res["year"].values
lams  = df_res["lambda"].values
ci_l  = df_res["ci_lo"].values
ci_h  = df_res["ci_hi"].values

# Panel 1: Clean Evolution of Lambda over Time (No text clutter inside plot)
ax1 = axes[0, 0]
ax1.fill_between(years, ci_l, ci_h, color="#f59e0b", alpha=0.25, label="95% Profile Likelihood CI")
ax1.plot(years, lams, marker="o", color="#d97706", lw=2.5, markersize=6, label="Geopolitical Weight λ(t)")
ax1.axhline(0.5, color="#64748b", ls="--", alpha=0.6, label="Balanced Gravity-Bloc (λ = 0.5)")
ax1.set_xlabel("Year", fontsize=10)
ax1.set_ylabel("Geopolitical Mixing Parameter (λ)", fontsize=10)
ax1.set_title("1. Global Geopolitical Bloc Weight λ (1965–2024)", fontweight="bold", fontsize=11)
ax1.set_ylim(0.15, 0.65)
ax1.grid(True, ls="--", alpha=0.35)
ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

# Panel 2: Country-by-Country World Trade Share (%) Evolution (Top 10 Powers)
ax2 = axes[0, 1]
country_colors = {
    "USA": "#1d4ed8", "CHN": "#dc2626", "DEU": "#047857", "JPN": "#7c3aed",
    "GBR": "#0284c7", "FRA": "#d97706", "ITA": "#059669", "KOR": "#ec4899",
    "IND": "#ea580c", "MEX": "#14b8a6"
}
for c in tracked_countries:
    ax2.plot(years, country_trade_evolution[c], marker="o", markersize=4.5, lw=2.0,
             color=country_colors.get(c, "#64748b"), label=f"{c} ({iso_to_name.get(c, c)[:12]})")
ax2.set_xlabel("Year", fontsize=10)
ax2.set_ylabel("Share of Total World Trade (%)", fontsize=10)
ax2.set_title("2. Country-Level Trade Share Evolution (1965–2024)", fontweight="bold", fontsize=11)
ax2.grid(True, ls="--", alpha=0.35)
ax2.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

# Panel 3: Country-Level Geopolitical vs Gravity Reliance in 2024 (Top 18 Economies)
ax3 = axes[1, 0]
top18 = latest_df_top.head(18).sort_values("geo_weight", ascending=True)
y_pos = np.arange(len(top18))
ax3.barh(y_pos, top18["grav_weight"], color="#3b82f6", alpha=0.85, label="Gravity Proximity Share (%)")
ax3.barh(y_pos, top18["geo_weight"], left=top18["grav_weight"], color="#f59e0b", alpha=0.85, label="Geopolitical Alliance Share (%)")
ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"{row['name'][:14]} ({row['iso']})" for _, row in top18.iterrows()], fontsize=8.5)
ax3.set_xlabel("Trade Network Decomposition (%)", fontsize=10)
ax3.set_title("3. Country Trade Decomposition: Gravity vs Geopolitical (2024)", fontweight="bold", fontsize=11)
ax3.set_xlim(0, 100)
ax3.grid(axis="x", ls="--", alpha=0.35)
ax3.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

# Panel 4: Inferred Geopolitical Latent Space S in 2024 (Clean Top Country Annotations)
ax4 = axes[1, 1]
sc4 = ax4.scatter(latest_S_hat[:, 0], latest_S_hat[:, 1],
                  c=np.log1p(latest_df_top["trade"]), cmap="plasma",
                  s=np.log1p(latest_df_top["trade"])*3.5 + 15,
                  alpha=0.85, edgecolors="k", lw=0.5)

# Label only top 12 global economies with clear annotations
top_12_iso = set(latest_df_top.head(12)["iso"])
for i in range(len(latest_df_top)):
    iso = latest_df_top.loc[i, "iso"]
    if iso in top_12_iso:
        ax4.annotate(iso, (latest_S_hat[i, 0], latest_S_hat[i, 1]),
                     textcoords="offset points", xytext=(4, 4),
                     fontsize=8.5, fontweight="bold", color="#0f172a",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

ax4.set_xlabel("Latent Geopolitical Dim 1", fontsize=10)
ax4.set_ylabel("Latent Geopolitical Dim 2", fontsize=10)
ax4.set_title("4. Inferred 2D Geopolitical Trade Bloc Space S (2024)", fontweight="bold", fontsize=11)
ax4.grid(True, ls="--", alpha=0.35)
cbar = plt.colorbar(sc4, ax=ax4)
cbar.set_label("Log Total Trade (2024)", fontsize=9)

plt.tight_layout()
plt.savefig("test_clean_refined_dashboard.png", dpi=150)
print("\nPlot saved successfully to test_clean_refined_dashboard.png")
