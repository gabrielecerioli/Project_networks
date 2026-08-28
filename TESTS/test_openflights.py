"""
Test script for OpenFlights network parameter inference.
"""

import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch

device = torch.device("cpu")

def test_openflights():
    # 1. Load airports
    airports_cols = ["airport_id", "name", "city", "country", "iata", "icao", "lat", "lon", "alt", "tz", "dst", "tz_db", "type", "source"]
    df_airports = pd.read_csv("../data/openflights/airports.dat", header=None, names=airports_cols, na_values="\\N")
    
    # Filter valid commercial airports with valid lat/lon
    df_airports = df_airports.dropna(subset=["airport_id", "lat", "lon"])
    df_airports["airport_id"] = df_airports["airport_id"].astype(int)
    print(f"Loaded {len(df_airports)} airports with valid coordinates")

    # 2. Load routes
    routes_cols = ["airline", "airline_id", "src_iata", "src_id", "dst_iata", "dst_id", "codeshare", "stops", "equipment"]
    df_routes = pd.read_csv("../data/openflights/routes.dat", header=None, names=routes_cols, na_values="\\N")
    df_routes = df_routes.dropna(subset=["src_id", "dst_id"])
    df_routes["src_id"] = pd.to_numeric(df_routes["src_id"], errors="coerce")
    df_routes["dst_id"] = pd.to_numeric(df_routes["dst_id"], errors="coerce")
    df_routes = df_routes.dropna(subset=["src_id", "dst_id"])
    df_routes["src_id"] = df_routes["src_id"].astype(int)
    df_routes["dst_id"] = df_routes["dst_id"].astype(int)
    print(f"Loaded {len(df_routes)} valid routes")

    # Count route frequency between airport pairs
    route_counts = df_routes.groupby(["src_id", "dst_id"]).size().reset_index(name="weight")

    # Get airport degrees (total connections) to select top commercial hubs (e.g. top 300)
    all_active_airports = set(route_counts["src_id"]).union(set(route_counts["dst_id"]))
    df_active = df_airports[df_airports["airport_id"].isin(all_active_airports)].copy()

    # Sum total traffic per airport
    src_deg = route_counts.groupby("src_id")["weight"].sum()
    dst_deg = route_counts.groupby("dst_id")["weight"].sum()
    total_traffic = src_deg.add(dst_deg, fill_value=0)
    df_active["traffic"] = df_active["airport_id"].map(total_traffic).fillna(0)

    # Select top N airports (e.g. N = 300 most connected global airports)
    N_HUBS = 300
    df_top = df_active.sort_values("traffic", ascending=False).head(N_HUBS).copy().reset_index(drop=True)
    top_ids = set(df_top["airport_id"])
    id_to_idx = {aid: i for i, aid in enumerate(df_top["airport_id"])}

    print(f"Selected Top {N_HUBS} Global Airport Hubs across {df_top['country'].nunique()} countries")
    print("Top 5 airports:", list(df_top[["iata", "city", "country", "traffic"]].head(5).itertuples(index=False)))

    # 3. 3D Coordinates on Earth Sphere (R = 6371 km)
    R_earth = 6371.0 # km
    lat_rad = np.radians(df_top["lat"].values)
    lon_rad = np.radians(df_top["lon"].values)
    
    x = R_earth * np.cos(lat_rad) * np.cos(lon_rad)
    y = R_earth * np.cos(lat_rad) * np.sin(lon_rad)
    z = R_earth * np.sin(lat_rad)
    X_3d = np.column_stack([x, y, z])

    # Great-circle / chordal distance matrix
    D_phys_np = cdist(X_3d, X_3d) # in km
    np.fill_diagonal(D_phys_np, 1e9)
    iu, ju = np.triu_indices(N_HUBS, k=1)

    # Build weighted adjacency matrix
    W_dir = np.zeros((N_HUBS, N_HUBS))
    for _, row in route_counts.iterrows():
        sid, did, w = int(row["src_id"]), int(row["dst_id"]), float(row["weight"])
        if sid in top_ids and did in top_ids:
            W_dir[id_to_idx[sid], id_to_idx[did]] += w

    W = 0.5 * (W_dir + W_dir.T)
    c_target = float(W.sum(axis=1).mean())

    print(f"Physical distance in km: min={D_phys_np.min():.1f} km, mean={D_phys_np[iu, ju].mean():.1f} km, max={D_phys_np[iu, ju].max():.1f} km")
    print(f"Network: {N_HUBS} nodes, {(W > 0).sum() // 2} links, mean strength c = {c_target:.2f}")

    r0 = float(2.0 * D_phys_np.min(axis=1).mean())
    log_r0 = float(np.log(r0))
    print(f"r0 prior center = {r0:.1f} km")

    # 4. Profile Likelihood Inference
    D_phys_t = torch.tensor(D_phys_np, dtype=torch.float32, device=device)
    W_obs_u = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
    iu_t = torch.tensor(iu, dtype=torch.long, device=device)
    ju_t = torch.tensor(ju, dtype=torch.long, device=device)
    eye_mask = 1.0 - torch.eye(N_HUBS, device=device)

    K_sp0 = np.exp(-D_phys_np / r0)
    np.fill_diagonal(K_sp0, 0.0)
    s_sp0 = K_sp0.sum(axis=1, keepdims=True) + 1e-12
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
    R_init = np.maximum(0.0, W - W_sp0)

    d_s = 2
    U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
    S_init = U[:, :d_s] * np.sqrt(sv[:d_s])
    S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0

    logit_lam = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    log_r = torch.tensor(log_r0, dtype=torch.float32, requires_grad=True, device=device)
    S_param = torch.tensor(S_init, dtype=torch.float32, requires_grad=True, device=device)

    # Adam warmup
    adam_opt = torch.optim.Adam([
        {"params": [logit_lam], "lr": 0.08},
        {"params": [log_r], "lr": 0.05},
        {"params": [S_param], "lr": 0.10}
    ])
    for _ in range(50):
        adam_opt.zero_grad()
        lam = torch.sigmoid(logit_lam)
        r = torch.exp(torch.clamp(log_r, log_r0 - 3.0, log_r0 + 3.0))
        K_sp = torch.exp(-D_phys_t / r) * eye_mask

        diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
        dist_sq = (diff ** 2).sum(-1) + 1e-12
        D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N_HUBS, device=device)
        K_soc = torch.exp(-D_soc / 1.0) * eye_mask

        W_raw = (1.0 - lam) * K_sp + lam * K_soc
        row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
        W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

        w_pred_u = W_pred[iu_t, ju_t]
        loss = (w_pred_u - W_obs_u * torch.log(w_pred_u + 1e-12)).sum()
        loss = loss + 0.5 * ((log_r - log_r0) / 2.0) ** 2
        loss.backward()
        adam_opt.step()

    # L-BFGS Polish
    lbfgs_opt = torch.optim.LBFGS(
        [logit_lam, log_r, S_param],
        lr=0.5,
        max_iter=45,
        history_size=10,
        line_search_fn="strong_wolfe"
    )

    def joint_closure():
        lbfgs_opt.zero_grad()
        lam = torch.sigmoid(logit_lam)
        r = torch.exp(torch.clamp(log_r, log_r0 - 3.0, log_r0 + 3.0))
        K_sp = torch.exp(-D_phys_t / r) * eye_mask

        diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
        dist_sq = (diff ** 2).sum(-1) + 1e-12
        D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N_HUBS, device=device)
        K_soc = torch.exp(-D_soc / 1.0) * eye_mask

        W_raw = (1.0 - lam) * K_sp + lam * K_soc
        row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
        W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

        w_pred_u = W_pred[iu_t, ju_t]
        loss = (w_pred_u - W_obs_u * torch.log(w_pred_u + 1e-12)).sum()
        loss = loss + 0.5 * ((log_r - log_r0) / 2.0) ** 2
        loss.backward()
        return loss

    lbfgs_opt.step(joint_closure)

    lam_hat = float(torch.sigmoid(logit_lam).item())
    r_hat = float(torch.exp(torch.clamp(log_r, log_r0 - 3.0, log_r0 + 3.0)).item())
    loss_center = joint_closure().item()

    # Fisher curvature
    delta_lam = 0.04
    profile_losses = [loss_center]
    for offset in [-delta_lam, delta_lam]:
        lam_target = np.clip(lam_hat + offset, 0.01, 0.99)
        lam_t = torch.tensor(float(lam_target), dtype=torch.float32, device=device)

        log_r_prof = log_r.clone().detach().requires_grad_(True)
        S_prof = S_param.clone().detach().requires_grad_(True)

        prof_opt = torch.optim.LBFGS([log_r_prof, S_prof], lr=0.5, max_iter=25, history_size=10, line_search_fn="strong_wolfe")

        def prof_closure():
            prof_opt.zero_grad()
            r = torch.exp(torch.clamp(log_r_prof, log_r0 - 3.0, log_r0 + 3.0))
            K_sp = torch.exp(-D_phys_t / r) * eye_mask

            diff = S_prof.unsqueeze(1) - S_prof.unsqueeze(0)
            dist_sq = (diff ** 2).sum(-1) + 1e-12
            D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N_HUBS, device=device)
            K_soc = torch.exp(-D_soc / 1.0) * eye_mask

            W_raw = (1.0 - lam_t) * K_sp + lam_t * K_soc
            row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
            W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

            w_pred_u = W_pred[iu_t, ju_t]
            loss = (w_pred_u - W_obs_u * torch.log(w_pred_u + 1e-12)).sum()
            loss = loss + 0.5 * ((log_r_prof - log_r0) / 2.0) ** 2
            loss.backward()
            return loss

        prof_opt.step(prof_closure)
        profile_losses.append(prof_closure().item())

    loss_minus = profile_losses[1]
    loss_plus = profile_losses[2]
    d2_nll = (loss_plus - 2.0 * loss_center + loss_minus) / (delta_lam ** 2)

    if d2_nll > 1e-3:
        lam_std = float(np.sqrt(1.0 / d2_nll))
    else:
        lam_std = 0.02

    ci_lo = float(np.clip(lam_hat - 1.96 * lam_std, 0.0, 1.0))
    ci_hi = float(np.clip(lam_hat + 1.96 * lam_std, 0.0, 1.0))

    print("\n" + "=" * 65)
    print("OPENFLIGHTS AIRPORT NETWORK INFERENCE RESULTS:")
    print("=" * 65)
    print(f"Estimated lambda_hat (Economic/Social weight) : {lam_hat:.4f} +/- {lam_std:.4f}")
    print(f"95% Confidence Interval                       : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Estimated Spatial Range r                     : {r_hat:.1f} km")
    print(f"Economic / Non-Local Hub Component            : {lam_hat * 100:.1f}%")
    print(f"Physical Geographical Proximity Component     : {(1.0 - lam_hat) * 100:.1f}%")
    print("=" * 65)

if __name__ == "__main__":
    test_openflights()
