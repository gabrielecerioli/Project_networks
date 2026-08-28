"""
Clean test script for World Commercial Trade Network parameter inference.
"""

import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch

device = torch.device("cpu")

def test_trade_network():
    # 1. Load Country coordinates
    coords_path = "../data/World_commercial_network/countries_codes_and_coordinates.csv"
    df_coords = pd.read_csv(coords_path)
    df_coords.columns = [c.strip().replace('"', '') for c in df_coords.columns]
    
    # Strip string columns
    for col in df_coords.columns:
        df_coords[col] = df_coords[col].astype(str).str.strip().str.replace('"', '')

    df_coords["lat"] = pd.to_numeric(df_coords["Latitude (average)"], errors="coerce")
    df_coords["lon"] = pd.to_numeric(df_coords["Longitude (average)"], errors="coerce")
    df_coords = df_coords.dropna(subset=["Alpha-3 code", "lat", "lon"]).copy()
    
    iso_to_lat = dict(zip(df_coords["Alpha-3 code"], df_coords["lat"]))
    iso_to_lon = dict(zip(df_coords["Alpha-3 code"], df_coords["lon"]))
    iso_to_name = dict(zip(df_coords["Alpha-3 code"], df_coords["Country"]))
    
    print(f"Loaded coordinates for {len(iso_to_lat)} countries")

    # 2. Load Trade flows
    trade_path = "../data/World_commercial_network/Trade_GPH0_1988_Export_2026Jan11.csv"
    df_trade = pd.read_csv(trade_path)
    df_trade.columns = [c.strip() for c in df_trade.columns]
    
    df_trade["src"] = df_trade["ReporterISO3"].astype(str).str.strip()
    df_trade["dst"] = df_trade["PartnerISO3"].astype(str).str.strip()
    df_trade["val"] = pd.to_numeric(df_trade["TradeValue"], errors="coerce").fillna(0.0)
    
    # Aggregate total bilateral trade value across all products
    trade_agg = df_trade.groupby(["src", "dst"])["val"].sum().reset_index()
    trade_agg = trade_agg[trade_agg["val"] > 0]
    
    # Countries present in coordinates
    valid_countries = sorted(list(set(trade_agg["src"]).union(set(trade_agg["dst"])).intersection(set(iso_to_lat.keys()))))
    
    # Filter to active trading nations
    country_trade = trade_agg.groupby("src")["val"].sum().add(trade_agg.groupby("dst")["val"].sum(), fill_value=0)
    
    df_countries = pd.DataFrame({
        "iso": valid_countries,
        "name": [iso_to_name[c] for c in valid_countries],
        "lat": [iso_to_lat[c] for c in valid_countries],
        "lon": [iso_to_lon[c] for c in valid_countries],
        "total_trade": [country_trade.get(c, 0.0) for c in valid_countries]
    })
    
    # Select top 120 trading nations
    N_TOP = min(120, len(df_countries[df_countries["total_trade"] > 0]))
    df_top = df_countries.sort_values("total_trade", ascending=False).head(N_TOP).reset_index(drop=True)
    top_iso = set(df_top["iso"])
    iso_to_idx = {iso: i for i, iso in enumerate(df_top["iso"])}
    
    print(f"Selected Top {N_TOP} Global Trading Nations in 1988")
    print("Top 5 Trading Nations:", list(df_top[["iso", "name", "total_trade"]].head(5).itertuples(index=False)))

    # 3. 3D Coordinates on Earth Sphere (R = 6371 km)
    R_EARTH = 6371.0 # km
    lat_rad = np.radians(df_top["lat"].values)
    lon_rad = np.radians(df_top["lon"].values)
    
    x = R_EARTH * np.cos(lat_rad) * np.cos(lon_rad)
    y = R_EARTH * np.cos(lat_rad) * np.sin(lon_rad)
    z = R_EARTH * np.sin(lat_rad)
    X_3d = np.column_stack([x, y, z])

    D_phys_np = cdist(X_3d, X_3d)
    np.fill_diagonal(D_phys_np, 1e9)
    iu, ju = np.triu_indices(N_TOP, k=1)

    # Build weighted bilateral trade matrix W
    W_dir = np.zeros((N_TOP, N_TOP))
    for _, row in trade_agg.iterrows():
        s, d, v = str(row["src"]), str(row["dst"]), float(row["val"])
        if s in top_iso and d in top_iso:
            W_dir[iso_to_idx[s], iso_to_idx[d]] += v

    # Symmetrize bilateral volume: total trade between i and j
    W_raw_val = 0.5 * (W_dir + W_dir.T)
    # Log-transform to stabilize heavy-tailed economic values
    W = np.log1p(W_raw_val)
    c_target = float(W.sum(axis=1).mean())

    print(f"Distance in km: min={D_phys_np.min():.1f} km, mean={D_phys_np[iu, ju].mean():.1f} km, max={D_phys_np[iu, ju].max():.1f} km")
    print(f"Network: {N_TOP} countries, {(W > 0).sum() // 2} active bilateral links, mean node strength c = {c_target:.2f}")

    r0 = float(2.0 * D_phys_np.min(axis=1).mean())
    log_r0 = float(np.log(r0))
    print(f"r0 prior center = {r0:.1f} km")

    # 4. Profile Likelihood Inference
    D_phys_t = torch.tensor(D_phys_np, dtype=torch.float32, device=device)
    W_obs_u = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
    iu_t = torch.tensor(iu, dtype=torch.long, device=device)
    ju_t = torch.tensor(ju, dtype=torch.long, device=device)
    eye_mask = 1.0 - torch.eye(N_TOP, device=device)

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
        D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N_TOP, device=device)
        K_soc = torch.exp(-D_soc / 1.0) * eye_mask

        W_raw = (1.0 - lam) * K_sp + lam * K_soc
        row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
        W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

        w_pred_u = W_pred[iu_t, ju_t]
        loss = (w_pred_u - W_obs_u * torch.log(w_pred_u + 1e-12)).sum()
        loss = loss + 0.5 * ((log_r - log_r0) / 2.0) ** 2
        loss.backward()
        adam_opt.step()

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
        D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N_TOP, device=device)
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
            D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N_TOP, device=device)
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
    print("WORLD COMMERCIAL TRADE NETWORK INFERENCE RESULTS:")
    print("=" * 65)
    print(f"Estimated lambda_hat (Geopolitical/Trade Bloc weight) : {lam_hat:.4f} +/- {lam_std:.4f}")
    print(f"95% Confidence Interval                              : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Estimated Spatial Trade Friction Scale r             : {r_hat:.1f} km")
    print(f"Geopolitical / Trade Agreement Component (λ)         : {lam_hat * 100:.1f}%")
    print(f"Geographical Distance Friction Component (1-λ)       : {(1.0 - lam_hat) * 100:.1f}%")
    print("=" * 65)

if __name__ == "__main__":
    test_trade_network()
