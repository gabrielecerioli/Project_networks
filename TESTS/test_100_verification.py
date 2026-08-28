"""
Test script for the 100-network Monte Carlo validation cell.
"""

import time
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
import torch

device = torch.device("cpu")

def generate_network_fast(N, lam, c, r, sigma=1.2, K=2, L=12.0, L_s=12.0, d_s=2, cluster_std=0.9, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, L, size=(N, 2))
    centers = rng.uniform(0.0, L_s, size=(K, d_s))
    cluster_id = rng.integers(0, K, size=N)
    S = centers[cluster_id] + rng.normal(0.0, cluster_std, size=(N, d_s))

    D_phys = cdist(X, X)
    D_soc = cdist(S, S)
    np.fill_diagonal(D_phys, np.inf)
    np.fill_diagonal(D_soc, np.inf)
    W_raw = (1.0 - lam) * np.exp(-D_phys / r) + lam * np.exp(-D_soc / sigma)

    W = c * W_raw / W_raw.sum(axis=1, keepdims=True)
    W = 0.5 * (W + W.T)
    return X, W, S, cluster_id

def infer_fast(X, W, c_target, d_s=2, sigma_gauge=1.0):
    t0 = time.time()
    N = len(X)
    D_phys_np = cdist(X, X)
    np.fill_diagonal(D_phys_np, 1e9)
    iu, ju = np.triu_indices(N, k=1)

    D_phys_t = torch.tensor(D_phys_np, dtype=torch.float32, device=device)
    W_obs_u = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
    iu_t = torch.tensor(iu, dtype=torch.long, device=device)
    ju_t = torch.tensor(ju, dtype=torch.long, device=device)
    eye_mask = 1.0 - torch.eye(N, device=device)

    r0 = float(2.0 * D_phys_np.min(axis=1).mean())
    log_r0 = float(np.log(r0))

    K_sp0 = np.exp(-D_phys_np / r0)
    np.fill_diagonal(K_sp0, 0.0)
    s_sp0 = K_sp0.sum(axis=1, keepdims=True)
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
    R_init = np.maximum(0.0, W - W_sp0)

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
    for _ in range(40):
        adam_opt.zero_grad()
        lam = torch.sigmoid(logit_lam)
        r = torch.exp(torch.clamp(log_r, log_r0 - 2.0, log_r0 + 2.0))
        K_sp = torch.exp(-D_phys_t / r) * eye_mask

        diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
        dist_sq = (diff ** 2).sum(-1) + 1e-12
        D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N, device=device)
        K_soc = torch.exp(-D_soc / sigma_gauge) * eye_mask

        W_raw = (1.0 - lam) * K_sp + lam * K_soc
        row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
        W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

        w_pred_u = W_pred[iu_t, ju_t]
        loss = (w_pred_u - W_obs_u * torch.log(w_pred_u + 1e-12)).sum()
        loss = loss + 0.5 * ((log_r - log_r0) / 1.5) ** 2
        loss.backward()
        adam_opt.step()

    lbfgs_opt = torch.optim.LBFGS(
        [logit_lam, log_r, S_param],
        lr=0.5,
        max_iter=35,
        history_size=10,
        line_search_fn="strong_wolfe"
    )

    def joint_closure():
        lbfgs_opt.zero_grad()
        lam = torch.sigmoid(logit_lam)
        r = torch.exp(torch.clamp(log_r, log_r0 - 2.0, log_r0 + 2.0))
        K_sp = torch.exp(-D_phys_t / r) * eye_mask

        diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
        dist_sq = (diff ** 2).sum(-1) + 1e-12
        D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N, device=device)
        K_soc = torch.exp(-D_soc / sigma_gauge) * eye_mask

        W_raw = (1.0 - lam) * K_sp + lam * K_soc
        row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
        W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

        w_pred_u = W_pred[iu_t, ju_t]
        loss = (w_pred_u - W_obs_u * torch.log(w_pred_u + 1e-12)).sum()
        loss = loss + 0.5 * ((log_r - log_r0) / 1.5) ** 2
        loss.backward()
        return loss

    lbfgs_opt.step(joint_closure)

    lam_opt = float(torch.sigmoid(logit_lam).item())
    r_opt = float(torch.exp(torch.clamp(log_r, log_r0 - 2.0, log_r0 + 2.0)).item())
    loss_center = joint_closure().item()

    delta_lam = 0.04
    profile_losses = [loss_center]

    for offset in [-delta_lam, delta_lam]:
        lam_target = np.clip(lam_opt + offset, 0.01, 0.99)
        lam_t = torch.tensor(float(lam_target), dtype=torch.float32, device=device)

        log_r_prof = log_r.clone().detach().requires_grad_(True)
        S_prof = S_param.clone().detach().requires_grad_(True)

        prof_opt = torch.optim.LBFGS([log_r_prof, S_prof], lr=0.5, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

        def prof_closure():
            prof_opt.zero_grad()
            r = torch.exp(torch.clamp(log_r_prof, log_r0 - 2.0, log_r0 + 2.0))
            K_sp = torch.exp(-D_phys_t / r) * eye_mask

            diff = S_prof.unsqueeze(1) - S_prof.unsqueeze(0)
            dist_sq = (diff ** 2).sum(-1) + 1e-12
            D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N, device=device)
            K_soc = torch.exp(-D_soc / sigma_gauge) * eye_mask

            W_raw = (1.0 - lam_t) * K_sp + lam_t * K_soc
            row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
            W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

            w_pred_u = W_pred[iu_t, ju_t]
            loss = (w_pred_u - W_obs_u * torch.log(w_pred_u + 1e-12)).sum()
            loss = loss + 0.5 * ((log_r_prof - log_r0) / 1.5) ** 2
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
        lam_std = 0.03

    ci_lo = float(np.clip(lam_opt - 1.96 * lam_std, 0.0, 1.0))
    ci_hi = float(np.clip(lam_opt + 1.96 * lam_std, 0.0, 1.0))

    return {
        "lam_hat": lam_opt,
        "lam_std": lam_std,
        "ci_95": [ci_lo, ci_hi],
        "r_hat": r_opt,
        "runtime": time.time() - t0
    }

if __name__ == "__main__":
    print("Testing 5 quick random runs...")
    rng = np.random.default_rng(42)
    for i in range(5):
        N = int(rng.choice([150, 200, 250]))
        lam = float(rng.uniform(0.1, 0.9))
        r = float(rng.uniform(0.9, 1.8))
        K = int(rng.integers(2, 6))
        c = float(rng.uniform(12.0, 18.0))
        X, W, S, cl = generate_network_fast(N, lam, c, r, K=K, seed=i)
        res = infer_fast(X, W, c_target=c)
        cov = res["ci_95"][0] <= lam <= res["ci_95"][1]
        print(f"Run {i+1}: N={N}, lam_true={lam:.3f} -> hat={res['lam_hat']:.3f} CI=[{res['ci_95'][0]:.3f}, {res['ci_95'][1]:.3f}] Cov={cov} r_true={r:.2f} r_hat={res['r_hat']:.2f} time={res['runtime']:.2f}s")
