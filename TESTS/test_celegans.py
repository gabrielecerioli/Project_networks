"""
Test script for C. elegans connectome parameter inference.
"""

import time
import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torch

device = torch.device("cpu")

def test_celegans_inference():
    # 1. Load GraphML
    G = nx.read_graphml("../data/c_elegans/celegans_connectome.graphml")
    print(f"Loaded C. elegans graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2. Extract node coordinates and cell names
    nodes = list(G.nodes())
    N = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    cell_names = [G.nodes[n].get("cell_name", n) for n in nodes]
    cell_classes = [G.nodes[n].get("cell_class", "Unknown") for n in nodes]
    ganglia = [G.nodes[n].get("ganglia", "Unknown") for n in nodes]

    # Extract 3D physical coordinates
    coords = []
    for n in nodes:
        x = float(G.nodes[n]["x"])
        y = float(G.nodes[n]["y"])
        z = float(G.nodes[n]["z"])
        coords.append([x, y, z])
    X = np.array(coords)

    # 3. Build symmetric weighted adjacency matrix W
    W_dir = np.zeros((N, N))
    for u, v, d in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        w = float(d.get("weight", 1.0))
        W_dir[i, j] += w

    # Symmetrize: undirected connectome
    W = 0.5 * (W_dir + W_dir.T)
    c_target = float(W.sum(axis=1).mean())

    print(f"Symmetric W: shape {W.shape}, non-zero edges: {(W > 0).sum() // 2}, mean node strength c = {c_target:.2f}")

    # 4. Compute 3D physical distances
    D_phys_np = cdist(X, X)
    np.fill_diagonal(D_phys_np, 1e9)
    iu, ju = np.triu_indices(N, k=1)

    print(f"Physical distance 3D: min={D_phys_np.min():.2f}, mean={D_phys_np[iu, ju].mean():.2f}, max={D_phys_np[iu, ju].max():.2f}")

    # Initial spatial scale prior
    r0 = float(2.0 * D_phys_np.min(axis=1).mean())
    log_r0 = float(np.log(r0))
    print(f"Initial spatial scale r0 = {r0:.2f}")

    # 5. Run Gauge-Invariant Profile Likelihood Inference
    D_phys_t = torch.tensor(D_phys_np, dtype=torch.float32, device=device)
    W_obs_u = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
    iu_t = torch.tensor(iu, dtype=torch.long, device=device)
    ju_t = torch.tensor(ju, dtype=torch.long, device=device)
    eye_mask = 1.0 - torch.eye(N, device=device)

    # SVD Residual Initialization
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

    # Adam Warmup (50 steps)
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
        D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N, device=device)
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
        D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N, device=device)
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

    # Curvature of Fisher
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
            D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N, device=device)
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

    print("\n" + "=" * 60)
    print("C. ELEGANS CONNECTOME INFERENCE RESULTS:")
    print("=" * 60)
    print(f"Estimated lambda_hat = {lam_hat:.4f} +/- {lam_std:.4f}")
    print(f"95% Confidence Int  = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Estimated spatial r = {r_hat:.2f} um")
    print(f"Spatial fraction    = {(1.0 - lam_hat)*100:.1f}%")
    print(f"Social/Functional   = {lam_hat*100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    test_celegans_inference()
