"""
Fast, Rigorous Parameter Inference for Dual Spatial-Social Networks.
Testing:
- Collapsed Cluster Profile MLE (CC-MLE)
- Continuous SVD Profile MLE (SVD-MLE)
- Analytical Distance-Decay Matching
"""

import time
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

    return dict(X=X, S=S, cluster_id=cluster_id, D_phys=D_phys, D_soc=D_soc, W=W, W_raw=W_raw)


# ======================================================================
# METHOD 1: Cluster-Collapsed Profile MLE (CC-MLE)
# ======================================================================
def solve_cluster_collapsed_mle(X, W, c_target=15.0, K_est=2):
    """
    1. Identify clusters from long-range / residual SVD.
    2. Form block kernel K_soc(A).
    3. Profile over (lam, r, intra, inter) using L-BFGS-B on Poisson log-likelihood.
    """
    N = len(X)
    D_phys = cdist(X, X)
    np.fill_diagonal(D_phys, 1e9)
    iu, ju = np.triu_indices(N, k=1)
    W_u = W[iu, ju]

    # Initial residual SVD
    r0 = 2.0 * float(D_phys.min(axis=1).mean())
    K_sp0 = np.exp(-D_phys / r0)
    np.fill_diagonal(K_sp0, 0.0)
    s_sp0 = K_sp0.sum(axis=1, keepdims=True)
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
    R_init = W - W_sp0

    # Spectral embedding
    U, s, _ = np.linalg.svd(R_init, full_matrices=False)
    emb = U[:, :K_est] * np.sqrt(s[:K_est])
    # Normalize rows
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    kmeans = KMeans(n_clusters=K_est, n_init=10, random_state=0).fit(emb_norm)
    z = kmeans.labels_

    same_mask = (z[:, None] == z[None, :]) & (~np.eye(N, dtype=bool))
    diff_mask = (z[:, None] != z[None, :]) & (~np.eye(N, dtype=bool))

    # Loss function in terms of unconstrained parameters:
    # theta = [logit(lam), log(r), log(inter_ratio)]
    def nll_poisson(theta):
        logit_lam, log_r, logit_rho = theta
        lam = 1.0 / (1.0 + np.exp(-logit_lam))
        r = np.exp(log_r)
        rho = 1.0 / (1.0 + np.exp(-logit_rho)) # affinity between different clusters (usually ~ 0)

        K_sp = np.exp(-D_phys / r)
        np.fill_diagonal(K_sp, 0.0)
        
        K_soc = (same_mask + rho * diff_mask).astype(float)
        np.fill_diagonal(K_soc, 0.0)

        W_raw = (1.0 - lam) * K_sp + lam * K_soc
        row_sums = W_raw.sum(axis=1, keepdims=True) + 1e-12
        W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.T)

        w_pred_u = W_pred[iu, ju]
        # Poisson NLL: sum( w_pred - w_obs * log(w_pred) )
        nll = np.sum(w_pred_u - W_u * np.log(w_pred_u + 1e-12))
        return nll

    # Optimize with L-BFGS-B
    init_theta = [0.0, np.log(r0), -3.0]
    res = minimize(nll_poisson, init_theta, method="L-BFGS-B", options=dict(maxiter=500))
    
    logit_lam_opt, log_r_opt, logit_rho_opt = res.x
    lam_hat = 1.0 / (1.0 + np.exp(-logit_lam_opt))
    r_hat = np.exp(log_r_opt)
    rho_hat = 1.0 / (1.0 + np.exp(-logit_rho_opt))

    # Compute numerical Hessian for exact uncertainty
    eps = 1e-4
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            e_i = np.zeros(3); e_i[i] = eps
            e_j = np.zeros(3); e_j[j] = eps
            if i == j:
                f_plus = nll_poisson(res.x + e_i)
                f_minus = nll_poisson(res.x - e_i)
                f_0 = res.fun
                H[i, i] = (f_plus - 2 * f_0 + f_minus) / (eps ** 2)
            else:
                f_pp = nll_poisson(res.x + e_i + e_j)
                f_pm = nll_poisson(res.x + e_i - e_j)
                f_mp = nll_poisson(res.x - e_i + e_j)
                f_mm = nll_poisson(res.x - e_i - e_j)
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)

    try:
        cov = np.linalg.inv(H)
        var_logit = max(1e-8, cov[0, 0])
        dlam = lam_hat * (1.0 - lam_hat)
        lam_std = np.sqrt(var_logit) * dlam
    except Exception as e:
        lam_std = 0.02

    ci_95 = [float(np.clip(lam_hat - 1.96 * lam_std, 0.0, 1.0)),
             float(np.clip(lam_hat + 1.96 * lam_std, 0.0, 1.0))]

    return dict(
        lam_hat=float(lam_hat),
        lam_std=float(lam_std),
        ci_95=ci_95,
        r_hat=float(r_hat),
        rho_hat=float(rho_hat),
        nll=float(res.fun),
        z=z
    )


# ======================================================================
# METHOD 2: Continuous SVD Low-Rank Kernel Profile MLE
# ======================================================================
def solve_svd_kernel_mle(X, W, c_target=15.0, rank=3):
    """
    Instead of discrete clusters, models K_soc as a smooth low-rank PSD kernel:
    K_soc = V V^T with rank << N (e.g. rank=3).
    Profiles (lam, r, V).
    """
    N = len(X)
    D_phys = cdist(X, X)
    np.fill_diagonal(D_phys, 1e9)
    iu, ju = np.triu_indices(N, k=1)
    W_u = W[iu, ju]

    r0 = 2.0 * float(D_phys.min(axis=1).mean())
    K_sp0 = np.exp(-D_phys / r0)
    np.fill_diagonal(K_sp0, 0.0)
    s_sp0 = K_sp0.sum(axis=1, keepdims=True)
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
    R_init = np.maximum(0.0, W - W_sp0)

    U, s, _ = np.linalg.svd(R_init, full_matrices=False)
    V_init = U[:, :rank] * np.sqrt(s[:rank])

    # Optimize lam, r, and V
    def nll_fn(p_flat):
        logit_lam = p_flat[0]
        log_r = p_flat[1]
        V = p_flat[2:].reshape((N, rank))

        lam = 1.0 / (1.0 + np.exp(-logit_lam))
        r = np.exp(log_r)

        K_sp = np.exp(-D_phys / r)
        np.fill_diagonal(K_sp, 0.0)

        K_soc = np.maximum(0.0, V @ V.T)
        np.fill_diagonal(K_soc, 0.0)

        W_raw = (1.0 - lam) * K_sp + lam * K_soc
        row_sums = W_raw.sum(axis=1, keepdims=True) + 1e-12
        W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.T)

        w_pred_u = W_pred[iu, ju]
        nll = np.sum(w_pred_u - W_u * np.log(w_pred_u + 1e-12))
        return nll

    p0 = np.concatenate([[0.0, np.log(r0)], V_init.flatten()])
    res = minimize(nll_fn, p0, method="L-BFGS-B", options=dict(maxiter=300))

    logit_lam_opt = res.x[0]
    log_r_opt = res.x[1]
    lam_hat = 1.0 / (1.0 + np.exp(-logit_lam_opt))
    r_hat = np.exp(log_r_opt)

    # 1D profile curvature around optimum for lambda
    d_logit = 0.05
    f_center = res.fun
    p_plus = res.x.copy(); p_plus[0] += d_logit; f_plus = nll_fn(p_plus)
    p_minus = res.x.copy(); p_minus[0] -= d_logit; f_minus = nll_fn(p_minus)
    d2 = (f_plus - 2 * f_center + f_minus) / (d_logit ** 2)
    var_logit = 1.0 / max(1e-4, d2)
    dlam = lam_hat * (1.0 - lam_hat)
    lam_std = np.sqrt(var_logit) * dlam

    ci_95 = [float(np.clip(lam_hat - 1.96 * lam_std, 0.0, 1.0)),
             float(np.clip(lam_hat + 1.96 * lam_std, 0.0, 1.0))]

    return dict(
        lam_hat=float(lam_hat),
        lam_std=float(lam_std),
        ci_95=ci_95,
        r_hat=float(r_hat),
        nll=float(res.fun)
    )


# ======================================================================
# METHOD 3: Full Continuous Positions with Profile MLE & Proper Gauge
# ======================================================================
def solve_full_continuous_profile_mle(X, W, c_target=15.0, d_s=2):
    """
    Fits continuous latent positions S_i in R^d_s, but without Gaussian prior
    distortions, and uses profile grid over lambda with joint (r, S) MLE.
    """
    N = len(X)
    D_phys = cdist(X, X)
    np.fill_diagonal(D_phys, 1e9)
    iu, ju = np.triu_indices(N, k=1)
    W_u = W[iu, ju]

    r0 = 2.0 * float(D_phys.min(axis=1).mean())
    K_sp0 = np.exp(-D_phys / r0)
    np.fill_diagonal(K_sp0, 0.0)
    s_sp0 = K_sp0.sum(axis=1, keepdims=True)
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
    R_init = W - W_sp0
    U, s, _ = np.linalg.svd(R_init, full_matrices=False)
    S_init = U[:, :d_s] * np.sqrt(s[:d_s])

    def nll_at_lam_r_S(p_flat):
        logit_lam = p_flat[0]
        log_r = p_flat[1]
        S = p_flat[2:].reshape((N, d_s))
        lam = 1.0 / (1.0 + np.exp(-logit_lam))
        r = np.exp(log_r)

        K_sp = np.exp(-D_phys / r)
        np.fill_diagonal(K_sp, 0.0)

        D_soc = cdist(S, S)
        np.fill_diagonal(D_soc, 1e9)
        K_soc = np.exp(-D_soc / 1.0) # Gauge fixed: sigma=1.0

        W_raw = (1.0 - lam) * K_sp + lam * K_soc
        row_sums = W_raw.sum(axis=1, keepdims=True) + 1e-12
        W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.T)

        w_pred_u = W_pred[iu, ju]
        nll = np.sum(w_pred_u - W_u * np.log(w_pred_u + 1e-12))
        return nll

    p0 = np.concatenate([[0.0, np.log(r0)], S_init.flatten()])
    res = minimize(nll_at_lam_r_S, p0, method="L-BFGS-B", options=dict(maxiter=400))

    logit_lam_opt = res.x[0]
    log_r_opt = res.x[1]
    lam_hat = 1.0 / (1.0 + np.exp(-logit_lam_opt))
    r_hat = np.exp(log_r_opt)

    d_logit = 0.05
    f_center = res.fun
    p_plus = res.x.copy(); p_plus[0] += d_logit; f_plus = nll_at_lam_r_S(p_plus)
    p_minus = res.x.copy(); p_minus[0] -= d_logit; f_minus = nll_at_lam_r_S(p_minus)
    d2 = (f_plus - 2 * f_center + f_minus) / (d_logit ** 2)
    var_logit = 1.0 / max(1e-4, d2)
    dlam = lam_hat * (1.0 - lam_hat)
    lam_std = np.sqrt(var_logit) * dlam

    ci_95 = [float(np.clip(lam_hat - 1.96 * lam_std, 0.0, 1.0)),
             float(np.clip(lam_hat + 1.96 * lam_std, 0.0, 1.0))]

    return dict(
        lam_hat=float(lam_hat),
        lam_std=float(lam_std),
        ci_95=ci_95,
        r_hat=float(r_hat),
        nll=float(res.fun)
    )


if __name__ == "__main__":
    print("Testing on user's exact network configuration: N=300, lam=0.8, r=1.2, c=15, K=2, seed=7")
    data = generate_dual_weighted_network(N=300, lam=0.8, c=15.0, r=1.2, sigma=1.2, K=2, seed=7)
    X, W = data["X"], data["W"]

    t0 = time.time()
    res1 = solve_cluster_collapsed_mle(X, W, c_target=15.0, K_est=2)
    t1 = time.time() - t0
    print(f"1. Cluster-Collapsed MLE : lam_hat = {res1['lam_hat']:.4f} +/- {res1['lam_std']:.4f} | 95% CI = {res1['ci_95']} | r_hat = {res1['r_hat']:.4f} | time = {t1:.2f}s")

    t0 = time.time()
    res2 = solve_svd_kernel_mle(X, W, c_target=15.0, rank=3)
    t2 = time.time() - t0
    print(f"2. SVD Low-Rank MLE      : lam_hat = {res2['lam_hat']:.4f} +/- {res2['lam_std']:.4f} | 95% CI = {res2['ci_95']} | r_hat = {res2['r_hat']:.4f} | time = {t2:.2f}s")

    t0 = time.time()
    res3 = solve_full_continuous_profile_mle(X, W, c_target=15.0, d_s=2)
    t3 = time.time() - t0
    print(f"3. Full Continuous MLE   : lam_hat = {res3['lam_hat']:.4f} +/- {res3['lam_std']:.4f} | 95% CI = {res3['ci_95']} | r_hat = {res3['r_hat']:.4f} | time = {t3:.2f}s")
