import time, torch, numpy as np, pandas as pd, networkx as nx, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

grid_dir = "../data/electrical_grids/GridKit ENTSO-E"
from test_station_level_grid import parse_buses, parse_lines

t0 = time.time()
df_buses = parse_buses(f"{grid_dir}/buses.csv")
df_lines = parse_lines(f"{grid_dir}/lines.csv")

# Filter active components
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

# Focus on High-Voltage Transmission Backbone (voltage >= 220 kV or all active lines in Continental GCC)
print(f"Total lines: {len(df_lines_st)}")
print("Voltage breakdown:\n", df_lines_st['voltage'].value_counts(dropna=False))

# Build graph of substations
G = nx.Graph()
for _, row in df_lines_st.iterrows():
    u, v = row['st0'], row['st1']
    v_val = float(row['voltage']) if not np.isnan(row['voltage']) else 220.0
    c_val = int(row['circuits']) if not np.isnan(row['circuits']) else 1
    w = c_val * (v_val / 100.0) # Weighted transmission capacity
    if G.has_edge(u, v):
        G[u][v]['weight'] += w
    else:
        G.add_edge(u, v, weight=w)

# Giant Connected Component
gcc_nodes = max(nx.connected_components(G), key=len)
print(f"\nGCC size: {len(gcc_nodes)} substations")

# Let's test on the primary transmission backbone (e.g. top N largest degree/capacity substations or Continental backbone)
# Let's test N = 1000, 2000, and full GCC
sub_nodes = sorted(list(gcc_nodes))
# Limit to N_nodes for fast benchmark
N_NODES = min(1500, len(sub_nodes)) # Or select top high-capacity nodes
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

print(f"Network Nodes: {N_NODES}, Active Links: {active_links}, Density: {density:.4f}")
print(f"Mean node strength c = {c_target:.2f}")
print(f"Direct connection line distance mean: {r0:.1f} km, range: [{D_phys[W > 0].min():.1f}, {D_phys[W > 0].max():.1f}] km")

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

logit_lam = torch.tensor(0.0, requires_grad=True, device=device)
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

# Profile likelihood CI
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
        D_soc = torch.sqrt((diff**2).sum(-1)+1e-12)+1e9*torch.eye(N_NODES, device=device)
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

print("\n" + "="*75)
print(f"ENTSO-E POWER GRID INFERENCE COMPLETE ({time.time() - t_opt:.2f}s)")
print("="*75)
print(f"• λ (Non-spatial / Functional weight) : {lam_hat:.4f} ± {lam_std:.4f}")
print(f"• 95% Confidence Interval             : [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"• r (Power line spatial reach)        : {r_hat:.1f} km (prior r0 = {r0:.1f} km)")
print(f"• Physical Gravity / Distance Share   : {(1.0 - lam_hat)*100:.1f}%")
print(f"• Non-spatial / Functional Share      : {lam_hat*100:.1f}%")
