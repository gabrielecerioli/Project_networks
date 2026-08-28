"""
PyTorch Vectorized Exact Maximum Likelihood / Profile Estimation
for Dual Spatial-Social Networks.

We test:
1. Exact Continuous Positions S in R^{N x d_s} via PyTorch L-BFGS / Adam
2. Gaussian Mixture / Cluster Centers Representation (C_k, cluster_std)
3. Direct Kernel Inversion with Exact Calibration
"""

import time
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import torch

# Set device and seeds
device = torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)

def generate_dual_weighted_network(
    N=300,
    lam=0.8,
    c=15.0,
    r=1.2,
    sigma=1.2,
    K=2,
    L=12.0,
    L_s=12.0,
    d_s=2,
    cluster_std=0.9,
    p_max=None,
    seed=7,
):
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

    if p_max is not None:
        W = np.minimum(W, p_max)

    return dict(X=X, S=S, cluster_id=cluster_id, D_phys=D_phys, D_soc=D_soc, W=W, W_raw=W_raw, centers=centers)


def solve_pytorch_continuous_mle(X, W, c_target=15.0, sigma_gauge=1.2, d_s=2, n_restarts=3):
    """
    Fits continuous latent positions S in R^{N x d_s} together with lambda and r.
    Uses SVD of residual as smart initialization.
    Optimizes using PyTorch L-BFGS with analytical autograd.
    """
    N = len(X)
    D_phys_np = cdist(X, X)
    np.fill_diagonal(D_phys_np, 1e9)
    iu, ju = np.triu_indices(N, k=1)
    
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    D_phys_t = torch.tensor(D_phys_np, dtype=torch.float32, device=device)
    W_obs_u = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
    iu_t = torch.tensor(iu, dtype=torch.long, device=device)
    ju_t = torch.tensor(ju, dtype=torch.long, device=device)

    # Initial residual SVD
    r0 = float(2.0 * D_phys_np.min(axis=1).mean())
    K_sp0 = np.exp(-D_phys_np / r0)
    np.fill_diagonal(K_sp0, 0.0)
    s_sp0 = K_sp0.sum(axis=1, keepdims=True)
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
    R_init = np.maximum(0.0, W - W_sp0)
    U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
    S_init = U[:, :d_s] * np.sqrt(sv[:d_s])
    # Scale to match sigma_gauge
    S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0

    best_loss = float("inf")
    best_res = None

    for restart in range(n_restarts):
        logit_lam = torch.tensor(0.0 + 0.3 * np.random.randn(), dtype=torch.float32, requires_grad=True, device=device)
        log_r = torch.tensor(np.log(r0) + 0.2 * np.random.randn(), dtype=torch.float32, requires_grad=True, device=device)
        
        # Jittered initial positions
        S_param = torch.tensor(S_init + 0.2 * np.random.randn(*S_init.shape), dtype=torch.float32, requires_grad=True, device=device)

        optimizer = torch.optim.LBFGS([logit_lam, log_r, S_param], lr=0.5, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            lam = torch.sigmoid(logit_lam)
            r = torch.exp(log_r)

            K_sp = torch.exp(-D_phys_t / r) * (1.0 - torch.eye(N, device=device))
            
            # Pairwise social distances
            diff = S_param.unsqueeze(1) - S_param.unsqueeze(0) # (N, N, d_s)
            dist_sq = (diff ** 2).sum(-1) + 1e-12
            D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N, device=device)
            K_soc = torch.exp(-D_soc / sigma_gauge) * (1.0 - torch.eye(N, device=device))

            W_raw = (1.0 - lam) * K_sp + lam * K_soc
            row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
            W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

            w_pred_u = W_pred[iu_t, ju_t]
            # Poisson Negative Log-Likelihood
            loss = (w_pred_u - W_obs_u * torch.log(w_pred_u + 1e-12)).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        final_loss = closure().item()

        if final_loss < best_loss:
            best_loss = final_loss
            lam_val = float(torch.sigmoid(logit_lam).item())
            r_val = float(torch.exp(log_r).item())
            best_res = dict(
                lam_hat=lam_val,
                r_hat=r_val,
                logit_lam=float(logit_lam.item()),
                log_r=float(log_r.item()),
                S_hat=S_param.detach().cpu().numpy(),
                loss=final_loss
            )

    return best_res


def profile_scan_exact(X, W, c_target=15.0, sigma_gauge=1.2, d_s=2, n_grid=21):
    """
    Profile Likelihood scan over lambda grid.
    For each lambda fixed, optimizes (r, S) via PyTorch L-BFGS.
    """
    N = len(X)
    D_phys_np = cdist(X, X)
    np.fill_diagonal(D_phys_np, 1e9)
    iu, ju = np.triu_indices(N, k=1)
    
    D_phys_t = torch.tensor(D_phys_np, dtype=torch.float32, device=device)
    W_obs_u = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
    iu_t = torch.tensor(iu, dtype=torch.long, device=device)
    ju_t = torch.tensor(ju, dtype=torch.long, device=device)

    r0 = float(2.0 * D_phys_np.min(axis=1).mean())
    K_sp0 = np.exp(-D_phys_np / r0)
    np.fill_diagonal(K_sp0, 0.0)
    s_sp0 = K_sp0.sum(axis=1, keepdims=True)
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
    R_init = np.maximum(0.0, W - W_sp0)
    U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
    S_init = U[:, :d_s] * np.sqrt(sv[:d_s])
    S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0

    grid_lam = np.linspace(0.05, 0.95, n_grid)
    nll_scores = []
    r_hats = []
    
    # Warm start S and log_r across grid
    current_S = S_init.copy()
    current_log_r = np.log(r0)

    for lam_val in grid_lam:
        lam_t = torch.tensor(float(lam_val), dtype=torch.float32, device=device)
        log_r = torch.tensor(current_log_r, dtype=torch.float32, requires_grad=True, device=device)
        S_param = torch.tensor(current_S, dtype=torch.float32, requires_grad=True, device=device)

        optimizer = torch.optim.LBFGS([log_r, S_param], lr=0.5, max_iter=150, history_size=15, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            r = torch.exp(log_r)
            K_sp = torch.exp(-D_phys_t / r) * (1.0 - torch.eye(N, device=device))
            
            diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
            dist_sq = (diff ** 2).sum(-1) + 1e-12
            D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N, device=device)
            K_soc = torch.exp(-D_soc / sigma_gauge) * (1.0 - torch.eye(N, device=device))

            W_raw = (1.0 - lam_t) * K_sp + lam_t * K_soc
            row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
            W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

            w_pred_u = W_pred[iu_t, ju_t]
            loss = (w_pred_u - W_obs_u * torch.log(w_pred_u + 1e-12)).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        final_loss = closure().item()
        nll_scores.append(final_loss)
        r_cur = float(torch.exp(log_r).item())
        r_hats.append(r_cur)
        
        # Warm start next step
        current_S = S_param.detach().cpu().numpy()
        current_log_r = float(log_r.item())

    nll_scores = np.array(nll_scores)
    r_hats = np.array(r_hats)

    best_idx = np.argmin(nll_scores)
    lam_best = grid_lam[best_idx]
    r_best = r_hats[best_idx]

    # Quadratic interpolation around minimum
    w_fit = 3
    i0 = max(0, best_idx - w_fit)
    i1 = min(len(grid_lam), best_idx + w_fit + 1)
    if i1 - i0 >= 3:
        p = np.polyfit(grid_lam[i0:i1], nll_scores[i0:i1], 2)
        if p[0] > 0:
            lam_min_quad = -p[1] / (2 * p[0])
            # Delta NLL = 1.92 corresponds to 95% profile likelihood CI (Wilks' theorem: 2 * delta NLL ~ chi^2_1, 1.96^2 / 2 = 1.9208)
            delta_nll_95 = 1.9208
            half_width = np.sqrt(delta_nll_95 / p[0])
            ci_95 = [float(np.clip(lam_min_quad - half_width, 0.0, 1.0)),
                     float(np.clip(lam_min_quad + half_width, 0.0, 1.0))]
            lam_std = float(half_width / 1.96)
            lam_hat = float(np.clip(lam_min_quad, 0.0, 1.0))
        else:
            lam_hat = float(lam_best)
            lam_std = 0.03
            ci_95 = [float(np.clip(lam_hat - 1.96*lam_std, 0.0, 1.0)), float(np.clip(lam_hat + 1.96*lam_std, 0.0, 1.0))]
    else:
        lam_hat = float(lam_best)
        lam_std = 0.03
        ci_95 = [float(np.clip(lam_hat - 1.96*lam_std, 0.0, 1.0)), float(np.clip(lam_hat + 1.96*lam_std, 0.0, 1.0))]

    return dict(
        lam_hat=lam_hat,
        lam_std=lam_std,
        ci_95=ci_95,
        r_hat=float(r_best),
        grid=grid_lam.tolist(),
        nll_scores=nll_scores.tolist(),
        r_hats=r_hats.tolist()
    )


if __name__ == "__main__":
    print("=== Testing Continuous PyTorch L-BFGS MLE ===")
    data = generate_dual_weighted_network(N=300, lam=0.8, c=15.0, r=1.2, sigma=1.2, K=2, seed=7)
    X, W = data["X"], data["W"]

    t0 = time.time()
    res_joint = solve_pytorch_continuous_mle(X, W, c_target=15.0, sigma_gauge=1.2, d_s=2, n_restarts=2)
    t_joint = time.time() - t0
    print(f"Joint MLE: lam_hat = {res_joint['lam_hat']:.4f} | r_hat = {res_joint['r_hat']:.4f} | loss = {res_joint['loss']:.2f} | time = {t_joint:.2f}s")

    t0 = time.time()
    res_prof = profile_scan_exact(X, W, c_target=15.0, sigma_gauge=1.2, d_s=2, n_grid=21)
    t_prof = time.time() - t0
    cov = res_prof['ci_95'][0] <= 0.8 <= res_prof['ci_95'][1]
    print(f"Profile Likelihood: lam_hat = {res_prof['lam_hat']:.4f} +/- {res_prof['lam_std']:.4f} | 95% CI = {res_prof['ci_95']} (covered: {cov}) | r_hat = {res_prof['r_hat']:.4f} | time = {t_prof:.2f}s")
    print("Grid scan NLL:", np.round(res_prof['nll_scores'], 1))
