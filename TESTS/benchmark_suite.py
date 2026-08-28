"""
Comprehensive Benchmark Suite for Dual Spatial-Social Network Parameter Inference.
Runs the exact Profile Likelihood solver on a diverse suite of 10+ synthetic networks
with varying:
- N (100 to 500)
- lambda_true (0.10 to 0.95)
- r_true (0.8 to 2.2)
- K (2 to 8 clusters)
- cluster_std (0.5 to 1.4)
- social dimensions d_s (2 to 3)

Measures:
- Parameter accuracy (MAE, RMSE for lambda and r)
- 95% CI coverage rate (target: ~95%)
- CI sharpness / width
- Computation runtime
- Saves detailed JSON metrics for the HTML dashboard
"""

import os
import json
import time
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
import torch

# Set device
device = torch.device("cpu")

def generate_dual_weighted_network(
    N=300,
    lam=0.5,
    c=15.0,
    r=1.2,
    sigma=1.2,
    K=4,
    L=12.0,
    L_s=12.0,
    d_s=2,
    cluster_std=0.9,
    p_max=None,
    seed=None,
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

    return dict(X=X, S=S, cluster_id=cluster_id, D_phys=D_phys, D_soc=D_soc, W=W, W_raw=W_raw)


def infer_dual_network_params(
    X,
    W,
    c_target=None,
    sigma_gauge=1.0,
    d_s=2,
    n_grid=21,
    lam_min=0.05,
    lam_max=0.95,
    max_iter_per_point=150,
    verbose=False
):
    """
    Robust, gauge-invariant Profile Likelihood Estimator for (lambda, r)
    in dual spatial-social networks.

    Parameters:
    -----------
    X : np.ndarray (N, 2)
        Known physical coordinates.
    W : np.ndarray (N, N)
        Observed symmetric weighted adjacency matrix.
    c_target : float or None
        Expected total node degree/weight (if None, estimated from W.sum(1).mean()).
    sigma_gauge : float
        Fixed social length scale gauge (default: 1.0).
    d_s : int
        Dimension of latent social space (default: 2).
    n_grid : int
        Number of grid points for profile scan over lambda.

    Returns:
    --------
    dict containing:
      - lam_hat : point estimate of lambda
      - lam_std : standard error of lambda
      - ci_95 : [lower, upper] 95% confidence interval
      - r_hat : point estimate of spatial interaction scale r
      - grid_lambdas : evaluated lambda values
      - profile_nll : negative log-likelihood values along profile
      - runtime : execution time in seconds
    """
    t0 = time.time()
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

    # Initial spatial scale prior center
    r0 = float(2.0 * D_phys_np.min(axis=1).mean())
    K_sp0 = np.exp(-D_phys_np / r0)
    np.fill_diagonal(K_sp0, 0.0)
    s_sp0 = K_sp0.sum(axis=1, keepdims=True)
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
    R_init = np.maximum(0.0, W - W_sp0)

    # Smart initialization via Truncated SVD of residual
    U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
    S_init = U[:, :d_s] * np.sqrt(sv[:d_s])
    S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0

    grid_lam = np.linspace(lam_min, lam_max, n_grid)
    nll_scores = []
    r_hats = []

    current_S = S_init.copy()
    current_log_r = np.log(r0)

    eye_mask = 1.0 - torch.eye(N, device=device)

    for lam_val in grid_lam:
        lam_t = torch.tensor(float(lam_val), dtype=torch.float32, device=device)
        log_r = torch.tensor(current_log_r, dtype=torch.float32, requires_grad=True, device=device)
        S_param = torch.tensor(current_S, dtype=torch.float32, requires_grad=True, device=device)

        optimizer = torch.optim.LBFGS(
            [log_r, S_param],
            lr=0.5,
            max_iter=max_iter_per_point,
            history_size=15,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            r = torch.exp(log_r)
            K_sp = torch.exp(-D_phys_t / r) * eye_mask

            diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
            dist_sq = (diff ** 2).sum(-1) + 1e-12
            D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(N, device=device)
            K_soc = torch.exp(-D_soc / sigma_gauge) * eye_mask

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

        current_S = S_param.detach().cpu().numpy()
        current_log_r = float(log_r.item())

        if verbose:
            print(f"  lam={lam_val:.3f} | NLL={final_loss:.2f} | r={r_cur:.3f}")

    nll_scores = np.array(nll_scores)
    r_hats = np.array(r_hats)

    best_idx = np.argmin(nll_scores)
    lam_best = grid_lam[best_idx]
    r_best = r_hats[best_idx]

    # Quadratic interpolation around minimum for sub-grid precision and profile likelihood CI
    w_fit = 3
    i0 = max(0, best_idx - w_fit)
    i1 = min(len(grid_lam), best_idx + w_fit + 1)
    if i1 - i0 >= 3:
        p = np.polyfit(grid_lam[i0:i1], nll_scores[i0:i1], 2)
        if p[0] > 0:
            lam_min_quad = -p[1] / (2 * p[0])
            delta_nll_95 = 1.9208 # chi2_1(0.95) / 2
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

    elapsed = time.time() - t0

    return dict(
        lam_hat=lam_hat,
        lam_std=lam_std,
        ci_95=ci_95,
        r_hat=float(r_best),
        grid_lambdas=grid_lam.tolist(),
        profile_nll=nll_scores.tolist(),
        r_hats=r_hats.tolist(),
        runtime=elapsed
    )


def run_full_benchmark():
    print("================================================================================")
    print("RUNNING EXTENSIVE 10-NETWORK BENCHMARK ON DUAL SPATIAL-SOCIAL PARAMETER INFERENCE")
    print("================================================================================")

    benchmark_configs = [
        # Network 1: The user's baseline network
        dict(id=1, N=300, lam=0.80, r=1.20, K=2, sigma=1.2, cluster_std=0.9, d_s=2, c=15.0, seed=7, desc="User baseline (lam=0.8, N=300, K=2)"),
        # Network 2: Low lambda (spatial dominance)
        dict(id=2, N=200, lam=0.15, r=1.50, K=3, sigma=1.2, cluster_std=0.8, d_s=2, c=15.0, seed=42, desc="Low lambda (lam=0.15, spatial dominant)"),
        # Network 3: Moderate lambda (equal balance)
        dict(id=3, N=250, lam=0.50, r=1.00, K=4, sigma=1.2, cluster_std=0.9, d_s=2, c=15.0, seed=101, desc="Balanced regime (lam=0.50, N=250, K=4)"),
        # Network 4: High lambda (strong social clustering)
        dict(id=4, N=200, lam=0.90, r=1.20, K=2, sigma=1.2, cluster_std=0.7, d_s=2, c=15.0, seed=202, desc="High lambda (lam=0.90, strong social)"),
        # Network 5: Large network N=500, lam=0.70
        dict(id=5, N=500, lam=0.70, r=1.20, K=4, sigma=1.2, cluster_std=0.9, d_s=2, c=15.0, seed=303, desc="Large network (N=500, lam=0.70)"),
        # Network 6: Small network N=100, lam=0.35, short spatial scale
        dict(id=6, N=100, lam=0.35, r=0.80, K=2, sigma=1.2, cluster_std=0.9, d_s=2, c=12.0, seed=404, desc="Small network (N=100, r=0.8, lam=0.35)"),
        # Network 7: Many clusters K=8, lam=0.60
        dict(id=7, N=350, lam=0.60, r=1.40, K=8, sigma=1.2, cluster_std=0.6, d_s=2, c=18.0, seed=505, desc="Many clusters (K=8, N=350, lam=0.60)"),
        # Network 8: 3D social space (d_s=3), lam=0.75
        dict(id=8, N=250, lam=0.75, r=1.20, K=3, sigma=1.2, cluster_std=0.9, d_s=3, c=15.0, seed=606, desc="3D latent social space (d_s=3, lam=0.75)"),
        # Network 9: Long spatial range (r=2.20), lam=0.40
        dict(id=9, N=300, lam=0.40, r=2.20, K=3, sigma=1.2, cluster_std=1.0, d_s=2, c=15.0, seed=707, desc="Long spatial range (r=2.20, lam=0.40)"),
        # Network 10: Extreme social dominance (lam=0.95)
        dict(id=10, N=250, lam=0.95, r=1.00, K=3, sigma=1.2, cluster_std=0.8, d_s=2, c=15.0, seed=808, desc="Extreme social (lam=0.95, N=250)"),
        # Network 11: Diffuse clusters (cluster_std=1.4), lam=0.65
        dict(id=11, N=300, lam=0.65, r=1.20, K=3, sigma=1.2, cluster_std=1.4, d_s=2, c=15.0, seed=909, desc="Diffuse social clusters (std=1.4, lam=0.65)"),
        # Network 12: Low density / small c (c=8.0), lam=0.55
        dict(id=12, N=200, lam=0.55, r=1.30, K=2, sigma=1.2, cluster_std=0.9, d_s=2, c=8.0, seed=999, desc="Low density strength (c=8.0, lam=0.55)")
    ]

    results = []

    for cfg in benchmark_configs:
        print(f"\n--- Benchmark #{cfg['id']}: {cfg['desc']} ---")
        print(f"    Params: N={cfg['N']}, lam_true={cfg['lam']}, r_true={cfg['r']}, K={cfg['K']}, d_s={cfg['d_s']}, c={cfg['c']}")

        data = generate_dual_weighted_network(
            N=cfg["N"], lam=cfg["lam"], c=cfg["c"], r=cfg["r"], sigma=cfg["sigma"],
            K=cfg["K"], d_s=cfg["d_s"], cluster_std=cfg["cluster_std"], seed=cfg["seed"]
        )

        res = infer_dual_network_params(
            data["X"], data["W"], c_target=cfg["c"], d_s=cfg["d_s"], n_grid=21
        )

        lam_true = cfg["lam"]
        r_true = cfg["r"]
        lam_hat = res["lam_hat"]
        r_hat = res["r_hat"]
        ci_lo, ci_hi = res["ci_95"]
        covered = (ci_lo <= lam_true <= ci_hi)
        err_lam = abs(lam_hat - lam_true)
        err_r = abs(r_hat - r_true)
        ci_width = ci_hi - ci_lo

        print(f"    Result : lam_hat = {lam_hat:.4f} +/- {res['lam_std']:.4f} | 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}] | Covered: {covered}")
        print(f"             r_hat   = {r_hat:.4f} (true={r_true:.2f}) | Error: |lam_err|={err_lam:.4f}, |r_err|={err_r:.4f} | Time: {res['runtime']:.2f}s")

        res_record = {
            "id": cfg["id"],
            "desc": cfg["desc"],
            "N": cfg["N"],
            "lam_true": lam_true,
            "lam_hat": lam_hat,
            "lam_std": res["lam_std"],
            "ci_95": [ci_lo, ci_hi],
            "ci_width": ci_width,
            "covered": covered,
            "r_true": r_true,
            "r_hat": r_hat,
            "err_lam": err_lam,
            "err_r": err_r,
            "K": cfg["K"],
            "d_s": cfg["d_s"],
            "c": cfg["c"],
            "runtime": res["runtime"],
            "grid_lambdas": res["grid_lambdas"],
            "profile_nll": res["profile_nll"]
        }
        results.append(res_record)

    # Compute overall summary metrics
    mae_lam = np.mean([r["err_lam"] for r in results])
    rmse_lam = np.sqrt(np.mean([r["err_lam"]**2 for r in results]))
    mae_r = np.mean([r["err_r"] for r in results])
    rmse_r = np.sqrt(np.mean([r["err_r"]**2 for r in results]))
    coverage_rate = np.mean([r["covered"] for r in results])
    avg_ci_width = np.mean([r["ci_width"] for r in results])
    avg_runtime = np.mean([r["runtime"] for r in results])

    print("\n" + "="*80)
    print("OVERALL BENCHMARK SUMMARY (12 Diverse Networks):")
    print("="*80)
    print(f"Lambda MAE          : {mae_lam:.4f}")
    print(f"Lambda RMSE         : {rmse_lam:.4f}")
    print(f"Spatial r MAE       : {mae_r:.4f}")
    print(f"Spatial r RMSE      : {rmse_r:.4f}")
    print(f"95% CI Coverage     : {coverage_rate*100:.1f}% ({sum(r['covered'] for r in results)}/{len(results)})")
    print(f"Average 95% CI Width: {avg_ci_width:.4f}")
    print(f"Average Runtime     : {avg_runtime:.2f}s per network")
    print("="*80)

    # Save to JSON
    summary_output = {
        "overall": {
            "n_networks": len(results),
            "mae_lambda": float(mae_lam),
            "rmse_lambda": float(rmse_lam),
            "mae_r": float(mae_r),
            "rmse_r": float(rmse_r),
            "coverage_rate_95": float(coverage_rate),
            "avg_ci_width": float(avg_ci_width),
            "avg_runtime": float(avg_runtime)
        },
        "records": results
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(summary_output, f, indent=2)

    print("Saved benchmark results to benchmark_results.json")
    return summary_output

if __name__ == "__main__":
    run_full_benchmark()
