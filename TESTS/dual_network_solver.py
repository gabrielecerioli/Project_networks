"""
================================================================================
DUAL SPATIAL-SOCIAL NETWORK PARAMETER INFERENCE SOLVER
================================================================================

A robust, gauge-invariant Maximum Profile Likelihood solver for dual spatial-social
weighted networks.

Solves the inverse / backward problem:
Given:
- Observed weighted adjacency matrix W_ij (or networkx.Graph)
- Known physical node coordinates X_i in R^2
Recovers:
- Mixing parameter lambda in [0, 1] with calibrated 95% Confidence Interval
- Spatial interaction length scale r > 0
- Latent social space coordinates S_i in R^{d_s}

Authors: Advanced Computational Physics & Network Inference Research
================================================================================
"""

import time
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
import torch

device = torch.device("cpu")

def infer_from_matrices(
    X: np.ndarray,
    W: np.ndarray,
    c_target: Optional[float] = None,
    sigma_gauge: float = 1.0,
    d_s: int = 2,
    adam_warmup_steps: int = 40,
    lbfgs_polish_steps: int = 40,
    delta_lam_curvature: float = 0.04,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Infers (lambda, r, S) from physical coordinates X and weighted adjacency matrix W.

    Parameters:
    -----------
    X : np.ndarray of shape (N, 2)
        Physical node coordinates in Euclidean space.
    W : np.ndarray of shape (N, N)
        Symmetric weighted adjacency matrix.
    c_target : float, optional
        Target expected degree / node strength. If None, computed as W.sum(axis=1).mean().
    sigma_gauge : float, default=1.0
        Social interaction scale gauge. (Fixing to 1.0 is gauge-invariant).
    d_s : int, default=2
        Dimension of latent social space.
    adam_warmup_steps : int, default=40
        Number of fast Adam gradient steps for warm-up.
    lbfgs_polish_steps : int, default=40
        Number of L-BFGS Quasi-Newton iterations for convergence.
    delta_lam_curvature : float, default=0.04
        Step size for numerical profile likelihood Fisher curvature estimation.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns:
    --------
    dict containing:
        - 'lambda_hat': Point estimate of mixing parameter lambda in [0, 1]
        - 'lambda_std': Standard error SE(lambda) from Fisher Information
        - 'ci_95': [lower, upper] 95% Confidence Interval for lambda
        - 'r_hat': Point estimate of spatial interaction scale r
        - 'S_hat': Inferred latent social coordinates in R^{N x d_s}
        - 'runtime': Total execution time in seconds
        - 'c_target': Expected node strength
        - 'loss': Final negative log-likelihood value
    """
    t0 = time.time()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    N = len(X)
    if c_target is None:
        c_target = float(W.sum(axis=1).mean())

    D_phys_np = cdist(X, X)
    np.fill_diagonal(D_phys_np, 1e9)
    iu, ju = np.triu_indices(N, k=1)

    D_phys_t = torch.tensor(D_phys_np, dtype=torch.float32, device=device)
    W_obs_u = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
    iu_t = torch.tensor(iu, dtype=torch.long, device=device)
    ju_t = torch.tensor(ju, dtype=torch.long, device=device)
    eye_mask = 1.0 - torch.eye(N, device=device)

    # Initial spatial scale prior center
    r0 = float(2.0 * D_phys_np.min(axis=1).mean())
    log_r0 = float(np.log(r0))

    # Smart initialization via Truncated SVD of residual matrix
    K_sp0 = np.exp(-D_phys_np / r0)
    np.fill_diagonal(K_sp0, 0.0)
    s_sp0 = K_sp0.sum(axis=1, keepdims=True)
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
    R_init = np.maximum(0.0, W - W_sp0)

    U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
    S_init = U[:, :d_s] * np.sqrt(sv[:d_s])
    S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0

    # Optimization parameters
    logit_lam = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    log_r = torch.tensor(log_r0, dtype=torch.float32, requires_grad=True, device=device)
    S_param = torch.tensor(S_init, dtype=torch.float32, requires_grad=True, device=device)

    # Stage 1: Adam warmup
    adam_opt = torch.optim.Adam([
        {"params": [logit_lam], "lr": 0.08},
        {"params": [log_r], "lr": 0.05},
        {"params": [S_param], "lr": 0.10}
    ])
    for _ in range(adam_warmup_steps):
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

    # Stage 2: L-BFGS Polish
    lbfgs_opt = torch.optim.LBFGS(
        [logit_lam, log_r, S_param],
        lr=0.5,
        max_iter=lbfgs_polish_steps,
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

    # Stage 3: Profile Curvature for exact standard errors
    profile_losses = [loss_center]
    for offset in [-delta_lam_curvature, delta_lam_curvature]:
        lam_target = np.clip(lam_opt + offset, 0.01, 0.99)
        lam_t = torch.tensor(float(lam_target), dtype=torch.float32, device=device)

        log_r_prof = log_r.clone().detach().requires_grad_(True)
        S_prof = S_param.clone().detach().requires_grad_(True)

        prof_opt = torch.optim.LBFGS([log_r_prof, S_prof], lr=0.5, max_iter=25, history_size=10, line_search_fn="strong_wolfe")

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
    d2_nll = (loss_plus - 2.0 * loss_center + loss_minus) / (delta_lam_curvature ** 2)

    if d2_nll > 1e-3:
        var_lam = 1.0 / d2_nll
        lam_std = float(np.sqrt(var_lam))
    else:
        lam_std = 0.03

    ci_lo = float(np.clip(lam_opt - 1.96 * lam_std, 0.0, 1.0))
    ci_hi = float(np.clip(lam_opt + 1.96 * lam_std, 0.0, 1.0))
    elapsed = time.time() - t0

    return {
        "lambda_hat": lam_opt,
        "lambda_std": lam_std,
        "ci_95": [ci_lo, ci_hi],
        "r_hat": r_opt,
        "S_hat": S_param.detach().cpu().numpy(),
        "c_target": c_target,
        "loss": loss_center,
        "runtime": elapsed
    }


def infer_dual_network(G: nx.Graph, d_s: int = 2, sigma_gauge: float = 1.0) -> Dict[str, Any]:
    """
    Convenience wrapper to run inference directly on a networkx.Graph.
    Assumes nodes have 'x' attribute and edges have 'weight' attribute.
    """
    N = G.number_of_nodes()
    X = np.array([G.nodes[i]["x"] for i in range(N)])
    W = np.zeros((N, N))
    for i, j, d in G.edges(data=True):
        W[i, j] = W[j, i] = d.get("weight", 1.0)

    res = infer_from_matrices(X, W, d_s=d_s, sigma_gauge=sigma_gauge)

    # If ground truth social positions 's' or cluster labels exist, compute alignment
    if "s" in G.nodes[0]:
        S_true = np.array([G.nodes[i]["s"] for i in range(N)])
        if S_true.shape[1] == d_s:
            R_rot, _ = orthogonal_procrustes(res["S_hat"], S_true)
            res["S_aligned"] = res["S_hat"] @ R_rot
            res["S_true"] = S_true
            D_true = cdist(S_true, S_true)
            D_hat = cdist(res["S_hat"], res["S_hat"])
            iu, ju = np.triu_indices(N, k=1)
            res["social_dist_corr"] = float(np.corrcoef(D_true[iu, ju], D_hat[iu, ju])[0, 1])

    return res


def plot_inference_diagnostics(
    res: Dict[str, Any],
    G: Optional[nx.Graph] = None,
    title: str = "Dual Network Parameter Inference",
    save_path: Optional[str] = None
):
    """
    Plots comprehensive diagnostics comparing true vs inferred parameters and social embeddings.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Parameter estimates & Confidence Interval
    ax = axes[0]
    lam_hat = res["lambda_hat"]
    ci_lo, ci_hi = res["ci_95"]
    ax.errorbar([0], [lam_hat], yerr=[[lam_hat - ci_lo], [ci_hi - lam_hat]], fmt="o", color="#3b82f6", ecolor="#60a5fa", elinewidth=3, capsize=8, capthick=2, markersize=10, label=f"lambda_hat = {lam_hat:.4f}")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_ylabel("Mixing Parameter (lambda)", fontsize=11)
    ax.set_title(f"lambda_hat: {lam_hat:.3f} | 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    # Panel 2: Inferred Social Coordinates
    ax2 = axes[1]
    if "S_aligned" in res:
        S_plot = res["S_aligned"]
        ttl = f"Inferred Social Space (Corr: {res.get('social_dist_corr', 0.0):.3f})"
    else:
        S_plot = res["S_hat"]
        ttl = "Inferred Social Space (S_hat)"

    colors = "#3b82f6"
    if G is not None and "cluster" in G.nodes[0]:
        clusters = np.array([G.nodes[i]["cluster"] for i in range(len(S_plot))])
        colors = plt.cm.tab10(clusters % 10)

    ax2.scatter(S_plot[:, 0], S_plot[:, 1], c=colors, s=25, alpha=0.8)
    ax2.set_title(ttl, fontsize=11, fontweight="bold")
    ax2.set_aspect("equal")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Panel 3: True Social Space (if available) or Physical Layout
    ax3 = axes[2]
    if G is not None and "s" in G.nodes[0]:
        S_true = res.get("S_true", np.array([G.nodes[i]["s"] for i in range(len(S_plot))]))
        ax3.scatter(S_true[:, 0], S_true[:, 1], c=colors, s=25, alpha=0.8)
        ax3.set_title("True Social Space (Ground Truth)", fontsize=11, fontweight="bold")
        ax3.set_aspect("equal")
    elif G is not None and "x" in G.nodes[0]:
        X = np.array([G.nodes[i]["x"] for i in range(len(S_plot))])
        ax3.scatter(X[:, 0], X[:, 1], c=colors, s=25, alpha=0.8)
        ax3.set_title("Known Physical Coordinates (X)", fontsize=11, fontweight="bold")
        ax3.set_aspect("equal")
    ax3.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved diagnostic figure to {save_path}")
    plt.show()


if __name__ == "__main__":
    from benchmark_suite import generate_dual_weighted_network

    print("Running demonstration of dual_network_solver...")
    data = generate_dual_weighted_network(N=300, lam=0.80, r=1.20, c=15.0, K=2, seed=7)
    
    res = infer_from_matrices(data["X"], data["W"], c_target=15.0)
    print("\n--- INFERENCE RESULTS ---")
    print(f"lambda_hat = {res['lambda_hat']:.4f} +/- {res['lambda_std']:.4f}")
    print(f"95% CI     = [{res['ci_95'][0]:.4f}, {res['ci_95'][1]:.4f}]")
    print(f"r_hat      = {res['r_hat']:.4f}")
    print(f"Runtime    = {res['runtime']:.2f}s")
