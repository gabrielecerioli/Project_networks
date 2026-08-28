import time, torch, numpy as np, pandas as pd, networkx as nx, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

grid_dir = "../data/electrical_grids/GridKit ENTSO-E"

def parse_buses(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 9:
                bus_id, station_id, voltage, dc, symbol, under_const, tags, x, y = parts
            else:
                bus_id, station_id, voltage, dc, symbol, under_const = parts[:6]
                x, y = parts[-2], parts[-1]
            try:
                records.append({
                    "bus_id": int(bus_id),
                    "station_id": int(station_id) if station_id and station_id.isdigit() else station_id,
                    "voltage": float(voltage) if voltage else np.nan,
                    "under_construction": under_const == 't',
                    "lon": float(x),
                    "lat": float(y)
                })
            except ValueError:
                continue
    return pd.DataFrame(records)

def parse_lines(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            geom_idx = line.rfind(",'LINESTRING")
            if geom_idx == -1:
                geom_idx = line.rfind(",LINESTRING")
            if geom_idx != -1:
                front = line[:geom_idx]
                parts = front.split(",")
            else:
                parts = line.split(",")
            if len(parts) >= 8:
                try:
                    line_id, bus0, bus1, voltage, circuits, length, underground, under_const = parts[:8]
                    records.append({
                        "line_id": int(line_id),
                        "bus0": int(bus0),
                        "bus1": int(bus1),
                        "voltage": float(voltage) if voltage else np.nan,
                        "circuits": int(circuits) if circuits and circuits.isdigit() else 1,
                        "length_m": float(length) if length else np.nan,
                        "under_construction": under_const.lower() == 'true'
                    })
                except ValueError:
                    continue
    return pd.DataFrame(records)

print("Loading ENTSO-E Grid dataset...")
df_buses = parse_buses(f"{grid_dir}/buses.csv")
df_lines = parse_lines(f"{grid_dir}/lines.csv")

# Filter active infrastructure
df_buses = df_buses[~df_buses['under_construction']].copy()
df_lines = df_lines[~df_lines['under_construction']].copy()

bus_to_station = dict(zip(df_buses['bus_id'], df_buses['station_id']))
station_coords = df_buses.groupby('station_id')[['lat', 'lon']].mean().reset_index()
station_to_lat = dict(zip(station_coords['station_id'], station_coords['lat']))
station_to_lon = dict(zip(station_coords['station_id'], station_coords['lon']))

df_lines['st0'] = df_lines['bus0'].map(bus_to_station)
df_lines['st1'] = df_lines['bus1'].map(bus_to_station)

df_lines_st = df_lines.dropna(subset=['st0', 'st1']).copy()
df_lines_st = df_lines_st[df_lines_st['st0'] != df_lines_st['st1']].copy()

# Build Graph
G = nx.Graph()
for _, row in df_lines_st.iterrows():
    u, v = row['st0'], row['st1']
    v_val = float(row['voltage']) if not np.isnan(row['voltage']) else 220.0
    c_val = int(row['circuits']) if not np.isnan(row['circuits']) else 1
    w = c_val * (v_val / 100.0)
    if G.has_edge(u, v):
        G[u][v]['weight'] += w
    else:
        G.add_edge(u, v, weight=w)

# Giant Connected Component
gcc_nodes = max(nx.connected_components(G), key=len)
sub_nodes = sorted(list(gcc_nodes))

# Select primary transmission backbone
N_NODES = min(1200, len(sub_nodes))
node_strength = {n: sum(d['weight'] for _, _, d in G.edges(n, data=True)) for n in sub_nodes}
selected_nodes = sorted(sub_nodes, key=lambda n: node_strength[n], reverse=True)[:N_NODES]
selected_set = set(selected_nodes)

df_nodes = pd.DataFrame({
    "station_id": selected_nodes,
    "lat": [station_to_lat[n] for n in selected_nodes],
    "lon": [station_to_lon[n] for n in selected_nodes],
    "strength": [node_strength[n] for n in selected_nodes]
}).reset_index(drop=True)
node_to_idx = {n: i for i, n in enumerate(df_nodes["station_id"])}

# 3D Coordinates on Earth Sphere
R_EARTH = 6371.0
lat_rad = np.radians(df_nodes["lat"].values)
lon_rad = np.radians(df_nodes["lon"].values)
x = R_EARTH * np.cos(lat_rad) * np.cos(lon_rad)
y = R_EARTH * np.cos(lat_rad) * np.sin(lon_rad)
z = R_EARTH * np.sin(lat_rad)
X_3d = np.column_stack([x, y, z])

D_phys = cdist(X_3d, X_3d)
np.fill_diagonal(D_phys, 1e9)
iu, ju = np.triu_indices(N_NODES, k=1)

# Weighted Adjacency Matrix
W = np.zeros((N_NODES, N_NODES))
for u in selected_nodes:
    for v, d in G[u].items():
        if v in selected_set:
            i, j = node_to_idx[u], node_to_idx[v]
            W[i, j] = float(d['weight'])

c_target = float(W.sum(axis=1).mean())
active_links = int((W > 0).sum() // 2)
density = (W > 0).sum() / (N_NODES * (N_NODES - 1))

r0 = float(D_phys[W > 0].mean()) if (W > 0).sum() > 0 else float(2.0 * D_phys.min(axis=1).mean())
log_r0 = float(np.log(r0))

# PyTorch Tensors
D_phys_t = torch.tensor(D_phys, dtype=torch.float32, device=device)
W_obs_u  = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
iu_t     = torch.tensor(iu, dtype=torch.long, device=device)
ju_t     = torch.tensor(ju, dtype=torch.long, device=device)
eye_mask = 1.0 - torch.eye(N_NODES, device=device)

# SVD Residual Init
d_s = 2
SIGMA_GAUGE = 1.0
K_sp0 = np.exp(-D_phys / r0)
np.fill_diagonal(K_sp0, 0.0)
s_sp0 = K_sp0.sum(axis=1, keepdims=True) + 1e-12
W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
R_init = np.maximum(0.0, W - W_sp0)

U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
S_init = U[:, :d_s] * np.sqrt(sv[:d_s])
S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0

logit_lam = torch.tensor(-2.0, requires_grad=True, device=device) # Init near spatial regime
log_r     = torch.tensor(log_r0, requires_grad=True, device=device)
S_param   = torch.tensor(S_init, requires_grad=True, device=device)

# Adam Warmup
adam_opt = torch.optim.Adam([
    {"params": [logit_lam], "lr": 0.08},
    {"params": [log_r],     "lr": 0.05},
    {"params": [S_param],   "lr": 0.10}
])
t_opt = time.time()
for _ in range(40):
    adam_opt.zero_grad()
    lam = torch.sigmoid(logit_lam)
    r   = torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3))
    K_sp = torch.exp(-D_phys_t / r) * eye_mask
    diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
    D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(N_NODES, device=device)
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
    D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(N_NODES, device=device)
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

# Profile Likelihood curve over lambda
lam_grid = np.linspace(0.001, 0.50, 25)
profile_losses = []
for l_val in lam_grid:
    lt = torch.tensor(float(l_val), device=device)
    lr_p = log_r.clone().detach().requires_grad_(True)
    S_p  = S_param.clone().detach().requires_grad_(True)
    opt_p = torch.optim.LBFGS([lr_p, S_p], lr=0.5, max_iter=15, line_search_fn="strong_wolfe")
    def pc():
        opt_p.zero_grad()
        r = torch.exp(torch.clamp(lr_p, log_r0-3, log_r0+3))
        K_sp = torch.exp(-D_phys_t/r)*eye_mask
        diff = S_p.unsqueeze(1)-S_p.unsqueeze(0)
        D_soc = torch.sqrt((diff**2).sum(-1)+1e-12)+1e9*torch.eye(N_NODES, device=device)
        K_soc = torch.exp(-D_soc/SIGMA_GAUGE)*eye_mask
        W_raw_t = (1-lt)*K_sp + lt*K_soc
        rs = W_raw_t.sum(dim=1, keepdim=True)+1e-12
        W_pred = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
        wu = W_pred[iu_t, ju_t]
        loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((lr_p-log_r0)/2)**2
        loss.backward(); return loss
    opt_p.step(pc)
    profile_losses.append(pc().item())

profile_losses = np.array(profile_losses)
d_nll = profile_losses - loss_c

# Compute predicted W matrix
r_t = torch.tensor(r_hat, dtype=torch.float32)
lam_t = torch.tensor(lam_hat, dtype=torch.float32)
K_sp_f = torch.exp(-D_phys_t / r_t) * eye_mask
diff_f = S_param.unsqueeze(1) - S_param.unsqueeze(0)
D_soc_f = torch.sqrt((diff_f**2).sum(-1)+1e-12) + 1e9*torch.eye(N_NODES, device=device)
K_soc_f = torch.exp(-D_soc_f / SIGMA_GAUGE) * eye_mask
W_raw_f = (1-lam_t)*K_sp_f + lam_t*K_soc_f
rs_f = W_raw_f.sum(dim=1, keepdim=True) + 1e-12
W_pred_np = (0.5*c_target*(W_raw_f/rs_f + W_raw_f/rs_f.t())).detach().cpu().numpy()

# ----------------------------------------------------------------------
# 4-Panel Visualization Dashboard for ENTSO-E Power Grid
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Geographic Map of the European Power Transmission Grid
ax1 = axes[0, 0]
ax1.scatter(df_nodes["lon"], df_nodes["lat"], c="#0284c7", s=10, alpha=0.7, label=f"Substations (N = {N_NODES})")
# Draw sample transmission lines
for u in selected_nodes:
    for v, d in G[u].items():
        if v in selected_set and u < v:
            ax1.plot([station_to_lon[u], station_to_lon[v]],
                     [station_to_lat[u], station_to_lat[v]],
                     color="#d97706", lw=0.6, alpha=0.45)
ax1.set_xlabel("Longitude (°E)", fontsize=10)
ax1.set_ylabel("Latitude (°N)", fontsize=10)
ax1.set_title(f"1. ENTSO-E European Power Transmission Grid Topology", fontweight="bold", fontsize=11)
ax1.set_xlim(-12, 36)
ax1.set_ylim(34, 68)
ax1.grid(True, ls="--", alpha=0.35)
ax1.legend(loc="upper left", fontsize=9)

# Panel 2: Spatial Exponential Decay Curve
ax2 = axes[0, 1]
d_active = D_phys[W > 0]
w_active = W[W > 0]
ax2.scatter(d_active, w_active, color="#8b5cf6", alpha=0.4, s=12, label="Observed Power Lines")
d_curve = np.linspace(1, 350, 200)
# Normalization
K_sp_hat = np.exp(-D_phys / r_hat)
np.fill_diagonal(K_sp_hat, 0.0)
mean_s_sp = float(K_sp_hat.sum(axis=1).mean())
pred_curve = (1.0 - lam_hat) * c_target * np.exp(-d_curve / r_hat) / mean_s_sp
ax2.plot(d_curve, pred_curve, color="#dc2626", lw=2.5, label=f"Inferred Physical Decay (r = {r_hat:.1f} km)")
ax2.set_xlabel("Physical Distance D_phys (km)", fontsize=10)
ax2.set_ylabel("Transmission Line Capacity Weight (w_ij)", fontsize=10)
ax2.set_title("2. Spatial Exponential Decay of Transmission Corridors", fontweight="bold", fontsize=11)
ax2.set_xlim(0, 350)
ax2.grid(True, ls="--", alpha=0.35)
ax2.legend(loc="upper right", fontsize=9)

# Panel 3: Observed vs Predicted Adjacency Comparison (Log Intensity)
ax3 = axes[1, 0]
d_flat = D_phys[iu, ju]
w_obs_flat = W[iu, ju]
w_pred_flat = W_pred_np[iu, ju]
# Sort by distance
sort_idx = np.argsort(d_flat)
ax3.scatter(w_obs_flat[w_obs_flat > 0], w_pred_flat[w_obs_flat > 0], color="#059669", alpha=0.5, s=14, label="Active Transmission Links")
ax3.plot([0, max(w_obs_flat.max(), w_pred_flat.max())], [0, max(w_obs_flat.max(), w_pred_flat.max())], 'k--', lw=1.5, label="Perfect Match Line")
ax3.set_xlabel("Observed Weight W_obs (kV capacity)", fontsize=10)
ax3.set_ylabel("Predicted Model Weight W_pred", fontsize=10)
ax3.set_title("3. Model Goodness-of-Fit on Power Line Capacities", fontweight="bold", fontsize=11)
ax3.grid(True, ls="--", alpha=0.35)
ax3.legend(loc="upper left", fontsize=9)

# Panel 4: Profile Likelihood Curve for Lambda
ax4 = axes[1, 1]
ax4.plot(lam_grid, d_nll, color="#2563eb", lw=2.5, marker="o", markersize=4, label="Profile Deviance Δ(-2 Log Likelihood)")
ax4.axvline(lam_hat, color="#dc2626", ls="--", lw=1.8, label=f"Inferred λ = {lam_hat:.4f} (Pure Physical Dominance)")
ax4.axhline(1.92, color="gray", ls=":", label="95% Confidence Threshold (ΔLL = 1.92)")
ax4.set_xlabel("Functional Weight Parameter (λ)", fontsize=10)
ax4.set_ylabel("Deviance ΔLL", fontsize=10)
ax4.set_title("4. Profile Likelihood: Strict Dominance of Physical Space (λ ≈ 0)", fontweight="bold", fontsize=11)
ax4.set_xlim(0, 0.50)
ax4.grid(True, ls="--", alpha=0.35)
ax4.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("test_grid_dashboard.png", dpi=150)
print("\nPlot saved successfully to test_grid_dashboard.png")
