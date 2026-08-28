"""
Comprehensive testbed for candidate parameter inference methods on dual spatial-social networks.

Candidate Methods to evaluate:
1. M1: Profile Nonlinear Least Squares / Log-Likelihood (Joint MLE over lambda, r, S)
2. M2: Collapsed Cluster Profile Likelihood (Infer cluster labels z, then joint MLE over lambda, r, centers)
3. M3: Direct Distance-Decay & Long-Range Moment Matching (Semi-parametric / Analytical)
4. M4: Full Bayesian HMC/NUTS via PyTorch / Pyro or Langevin Dynamics
5. M5: Structured Variational Inference with Proper Gauging and cluster priors
"""

import time
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.optimize import minimize, curve_fit
from sklearn.cluster import SpectralClustering, KMeans
import torch

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

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

    w_sp = (1.0 - lam) * np.exp(-D_phys / r)
    w_so = lam * np.exp(-D_soc / sigma)

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        G.nodes[i]["x"] = X[i]
        G.nodes[i]["s"] = S[i]
        G.nodes[i]["cluster"] = int(cluster_id[i])

    iu, ju = np.triu_indices(N, k=1)
    for i, j in zip(iu, ju):
        w_ij = W[i, j]
        t = "social" if w_so[i, j] > w_sp[i, j] else "spatial"
        G.add_edge(int(i), int(j),
                   weight=float(w_ij),
                   d_phys=float(D_phys[i, j]),
                   d_soc=float(D_soc[i, j]),
                   type=str(t))

    return G, dict(X=X, S=S, cluster_id=cluster_id, D_phys=D_phys, D_soc=D_soc, W=W, W_raw=W_raw, centers=centers)


# ======================================================================
# METHOD 1: Fast Analytical / Semi-Parametric Moment Inversion (M1)
# ======================================================================
def method_moment_inversion(X, W, c_target=15.0, K_est=2):
    """
    Exploits the mathematical fact that at long physical distances (D_phys >> r),
    the spatial kernel is ~0, so intra-cluster long-range edges directly reflect
    the social strength (lambda), while short-range decay reflects (1-lambda) and r.
    """
    N = len(X)
    D_phys = cdist(X, X)
    np.fill_diagonal(D_phys, 1e9)
    iu, ju = np.triu_indices(N, k=1)

    # 1. Estimate social clusters using spectral clustering on long-range edges or residual
    # Long range: D_phys > median(D_phys)
    med_d = np.median(D_phys[iu, ju])
    W_long = W.copy()
    W_long[D_phys < med_d] = 0.0
    
    # SVD / Spectral on W_long
    U, s, _ = np.linalg.svd(W_long, full_matrices=False)
    features = U[:, :K_est] * np.sqrt(s[:K_est])
    # Normalize rows
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-9
    features_norm = features / norms
    
    kmeans = KMeans(n_clusters=K_est, n_init=10, random_state=0).fit(features_norm)
    labels = kmeans.labels_

    # 2. Estimate r and lambda from the empirical distance-decay profile
    # For pairs in different clusters, social kernel is ~0, so W_ij is pure spatial!
    diff_cluster_mask = (labels[iu] != labels[ju])
    same_cluster_mask = (labels[iu] == labels[ju])

    d_diff = D_phys[iu, ju][diff_cluster_mask]
    w_diff = W[iu, ju][diff_cluster_mask]

    # Fit exponential decay on diff-cluster pairs: w(d) = A * exp(-d / r)
    # We can bin by distance
    bins = np.linspace(d_diff.min(), np.percentile(d_diff, 90), 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = []
    bin_counts = []
    for b_lo, b_hi in zip(bins[:-1], bins[1:]):
        mask = (d_diff >= b_lo) & (d_diff < b_hi)
        if mask.sum() > 5:
            bin_means.append(np.median(w_diff[mask]))
            bin_counts.append(mask.sum())
        else:
            bin_means.append(np.nan)
            bin_counts.append(0)

    valid = ~np.isnan(bin_means)
    x_fit = bin_centers[valid]
    y_fit = np.array(bin_means)[valid]

    def exp_func(d, A, r_val):
        return A * np.exp(-d / r_val)

    try:
        popt, pcov = curve_fit(exp_func, x_fit, y_fit, p0=[y_fit[0] if len(y_fit)>0 else 0.1, 1.2], bounds=(0, [10.0, 10.0]))
        A_est, r_est = popt
        r_std = np.sqrt(np.diag(pcov))[1]
    except Exception as e:
        A_est, r_est, r_std = 0.1, 1.2, 0.5

    # With r_est and cluster labels, we can compute expected row sums and solve for lambda
    # For intra-cluster long-range: w_same(d >> r) ~ B
    d_same = D_phys[iu, ju][same_cluster_mask]
    w_same = W[iu, ju][same_cluster_mask]
    long_same_mask = d_same > 3.0 * r_est
    if long_same_mask.sum() > 10:
        B_est = np.median(w_same[long_same_mask])
    else:
        B_est = np.median(w_same)

    # Relative ratio between spatial peak A and social baseline B
    # A ~ (1-lam)/s_avg, B ~ lam/s_avg -> lam / (1-lam) ~ B / A
    ratio = B_est / (A_est + 1e-12)
    lam_est = ratio / (1.0 + ratio)
    lam_est = float(np.clip(lam_est, 0.01, 0.99))

    return dict(lam_hat=lam_est, r_hat=r_est, r_std=r_std, labels=labels)


# ======================================================================
# METHOD 2: Exact Forward Model Profile Likelihood / NLS (M2)
# ======================================================================
def method_profile_likelihood(X, W, c_target=15.0, K_est=2, n_restarts=5):
    """
    Direct Maximum Likelihood / Least Squares over (lambda, r, S)
    using the EXACT forward model generator formulation, with
    gradient-based optimization in PyTorch with multiple restarts.
    """
    N = len(X)
    iu, ju = np.triu_indices(N, k=1)
    W_obs_t = torch.tensor(W[iu, ju], dtype=torch.float32)
    D_phys_t = torch.tensor(cdist(X, X), dtype=torch.float32)
    D_phys_t.fill_diagonal_(1e9)
    iu_t, ju_t = torch.tensor(iu), torch.tensor(ju)

    # Initialize S from residual spectral embedding
    r0 = 2.0 * float(D_phys_t.min(dim=1).values.mean())
    K_sp0 = torch.exp(-D_phys_t / r0)
    K_sp0.fill_diagonal_(0.0)
    s_sp0 = K_sp0.sum(dim=1, keepdim=True)
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.t())
    R_init = torch.tensor(W, dtype=torch.float32) - W_sp0
    
    U, s_svd, _ = torch.linalg.svd(R_init)
    S_init = U[:, :2] * torch.sqrt(s_svd[:2])
    S_init = (S_init - S_init.mean(dim=0)) / (S_init.std() + 1e-9)

    best_loss = float("inf")
    best_res = None

    # Sweep profile on lambda or optimize jointly
    for restart in range(n_restarts):
        # Parameters
        logit_lam = torch.tensor(0.0 + 0.5 * np.random.randn(), requires_grad=True)
        log_r = torch.tensor(np.log(r0) + 0.3 * np.random.randn(), requires_grad=True)
        log_sigma = torch.tensor(0.0, requires_grad=False) # Gauge fixed
        S_param = (S_init + 0.1 * torch.randn_like(S_init)).clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([logit_lam, log_r, S_param], lr=0.03)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500)

        for step in range(1500):
            optimizer.zero_grad()
            lam = torch.sigmoid(logit_lam)
            r_cur = torch.exp(log_r)
            
            # Pairwise social distances
            D_soc = torch.cdist(S_param, S_param)
            D_soc = D_soc + 1e9 * torch.eye(N)
            
            K_sp = torch.exp(-D_phys_t / r_cur)
            K_sp = K_sp * (1.0 - torch.eye(N))
            
            K_soc = torch.exp(-D_soc / 1.0)
            K_soc = K_soc * (1.0 - torch.eye(N))

            W_raw = (1.0 - lam) * K_sp + lam * K_soc
            row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
            W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

            # Objective: Poisson Log-Likelihood or Weighted MSE
            w_pred_u = W_pred[iu_t, ju_t]
            # Poisson NLL:
            loss = -(W_obs_t * torch.log(w_pred_u + 1e-12) - w_pred_u).sum()

            loss.backward()
            optimizer.step()
            scheduler.step()

        cur_loss = loss.item()
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_res = dict(
                lam_hat=float(torch.sigmoid(logit_lam).item()),
                r_hat=float(torch.exp(log_r).item()),
                S_hat=S_param.detach().cpu().numpy(),
                loss=cur_loss
            )

    return best_res


# ======================================================================
# METHOD 3: Profile Profile Grid with Hessian Uncertainty (M3)
# ======================================================================
def method_profile_hessian(X, W, c_target=15.0, K_est=2, n_grid=21):
    """
    Profile Likelihood scan over lambda grid.
    For each lambda in [0, 1], optimizes (r, S).
    Then fits a parabola near the maximum to obtain Fisher Information / Hessian variance.
    """
    N = len(X)
    iu, ju = np.triu_indices(N, k=1)
    W_obs_t = torch.tensor(W[iu, ju], dtype=torch.float32)
    D_phys_t = torch.tensor(cdist(X, X), dtype=torch.float32)
    D_phys_t.fill_diagonal_(1e9)
    iu_t, ju_t = torch.tensor(iu), torch.tensor(ju)

    r0 = 2.0 * float(D_phys_t.min(dim=1).values.mean())
    K_sp0 = torch.exp(-D_phys_t / r0)
    K_sp0.fill_diagonal_(0.0)
    s_sp0 = K_sp0.sum(dim=1, keepdim=True)
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.t())
    R_init = torch.tensor(W, dtype=torch.float32) - W_sp0
    U, s_svd, _ = torch.linalg.svd(R_init)
    S_init = U[:, :2] * torch.sqrt(s_svd[:2])
    S_init = (S_init - S_init.mean(dim=0)) / (S_init.std() + 1e-9)

    grid_lam = np.linspace(0.05, 0.95, n_grid)
    profile_nll = []
    r_hats = []

    for lam_val in grid_lam:
        lam_t = torch.tensor(float(lam_val))
        log_r = torch.tensor(np.log(r0), requires_grad=True)
        S_param = S_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([log_r, S_param], lr=0.03)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

        for step in range(1000):
            optimizer.zero_grad()
            r_cur = torch.exp(log_r)
            D_soc = torch.cdist(S_param, S_param) + 1e9 * torch.eye(N)
            K_sp = torch.exp(-D_phys_t / r_cur) * (1.0 - torch.eye(N))
            K_soc = torch.exp(-D_soc / 1.0) * (1.0 - torch.eye(N))

            W_raw = (1.0 - lam_t) * K_sp + lam_t * K_soc
            row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
            W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

            w_pred_u = W_pred[iu_t, ju_t]
            loss = -(W_obs_t * torch.log(w_pred_u + 1e-12) - w_pred_u).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

        profile_nll.append(loss.item())
        r_hats.append(torch.exp(log_r).item())

    profile_nll = np.array(profile_nll)
    r_hats = np.array(r_hats)

    best_idx = np.argmin(profile_nll)
    lam_hat_grid = grid_lam[best_idx]
    r_hat_grid = r_hats[best_idx]

    # Quadratic fit around minimum to get curvature (Fisher information)
    fit_window = 5
    start_i = max(0, best_idx - fit_window // 2)
    end_i = min(len(grid_lam), start_i + fit_window)
    if end_i - start_i < 3:
        start_i = max(0, end_i - 3)
    
    xs = grid_lam[start_i:end_i]
    ys = profile_nll[start_i:end_i]
    poly = np.polyfit(xs, ys, 2)
    a, b, c_poly = poly
    if a > 0:
        lam_hat_quad = -b / (2 * a)
        lam_var = 1.0 / a  # inverse Fisher curvature
        lam_std = np.sqrt(max(1e-6, lam_var))
    else:
        lam_hat_quad = lam_hat_grid
        lam_std = 0.05

    lam_hat = float(np.clip(lam_hat_quad, 0.0, 1.0))
    ci_95 = [float(np.clip(lam_hat - 1.96 * lam_std, 0.0, 1.0)),
             float(np.clip(lam_hat + 1.96 * lam_std, 0.0, 1.0))]

    return dict(
        lam_hat=lam_hat,
        lam_std=float(lam_std),
        ci_95=ci_95,
        r_hat=float(r_hat_grid),
        grid=grid_lam.tolist(),
        profile_nll=profile_nll.tolist()
    )


# ======================================================================
# METHOD 4: Block / Cluster-Collapsed Profile Estimator (M4)
# ======================================================================
def method_cluster_collapsed_mle(X, W, c_target=15.0, K_est=2):
    """
    Collapses the N continuous social positions S_i into K discrete clusters.
    1. Clusters nodes into K groups using spectral clustering on residual.
    2. Models social kernel as block matrix: K_soc_{ij} = delta(z_i, z_j) or continuous cluster centers.
    3. Jointly estimates (lambda, r, intra_decay, inter_decay).
    Eliminates the N-dimensional overfitting completely!
    """
    N = len(X)
    D_phys = cdist(X, X)
    np.fill_diagonal(D_phys, 1e9)
    iu, ju = np.triu_indices(N, k=1)

    # Initial residual SVD
    r0 = 2.0 * float(D_phys.min(axis=1).mean())
    K_sp0 = np.exp(-D_phys / r0)
    np.fill_diagonal(K_sp0, 0.0)
    s_sp0 = K_sp0.sum(axis=1, keepdims=True)
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
    R_init = W - W_sp0

    # Spectral clustering on R_init
    U, s, _ = np.linalg.svd(R_init, full_matrices=False)
    emb = U[:, :K_est] * np.sqrt(s[:K_est])
    kmeans = KMeans(n_clusters=K_est, n_init=10, random_state=0).fit(emb)
    z = kmeans.labels_

    same_mask = (z[:, None] == z[None, :]) & (~np.eye(N, dtype=bool))
    diff_mask = (z[:, None] != z[None, :]) & (~np.eye(N, dtype=bool))

    D_phys_t = torch.tensor(D_phys, dtype=torch.float32)
    same_t = torch.tensor(same_mask, dtype=torch.float32)
    diff_t = torch.tensor(diff_mask, dtype=torch.float32)
    W_obs_t = torch.tensor(W[iu, ju], dtype=torch.float32)
    iu_t, ju_t = torch.tensor(iu), torch.tensor(ju)

    # Optimize lambda, r, and between-cluster decay rho in [0, 1]
    logit_lam = torch.tensor(0.0, requires_grad=True)
    log_r = torch.tensor(np.log(r0), requires_grad=True)
    logit_rho = torch.tensor(-2.0, requires_grad=True) # between cluster affinity

    optimizer = torch.optim.Adam([logit_lam, log_r, logit_rho], lr=0.04)
    for step in range(1200):
        optimizer.zero_grad()
        lam = torch.sigmoid(logit_lam)
        r = torch.exp(log_r)
        rho = torch.sigmoid(logit_rho)

        K_sp = torch.exp(-D_phys_t / r) * (1.0 - torch.eye(N))
        K_soc = (same_t + rho * diff_t) * (1.0 - torch.eye(N))

        W_raw = (1.0 - lam) * K_sp + lam * K_soc
        row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
        W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())

        w_pred_u = W_pred[iu_t, ju_t]
        loss = -(W_obs_t * torch.log(w_pred_u + 1e-12) - w_pred_u).sum()
        loss.backward()
        optimizer.step()

    # Compute numerical Hessian at optimum for exact standard errors
    with torch.no_grad():
        lam_opt = torch.sigmoid(logit_lam).item()
        r_opt = torch.exp(log_r).item()

    # Evaluate Hessian of loss w.r.t (logit_lam, log_r)
    def loss_fn(p):
        lam = torch.sigmoid(p[0])
        r = torch.exp(p[1])
        rho = torch.sigmoid(p[2])
        K_sp = torch.exp(-D_phys_t / r) * (1.0 - torch.eye(N))
        K_soc = (same_t + rho * diff_t) * (1.0 - torch.eye(N))
        W_raw = (1.0 - lam) * K_sp + lam * K_soc
        row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
        W_pred = 0.5 * c_target * (W_raw / row_sums + W_raw / row_sums.t())
        w_pred_u = W_pred[iu_t, ju_t]
        return -(W_obs_t * torch.log(w_pred_u + 1e-12) - w_pred_u).sum()

    p_opt = torch.tensor([logit_lam.item(), log_r.item(), logit_rho.item()], requires_grad=True)
    hessian = torch.autograd.functional.hessian(loss_fn, p_opt)
    
    try:
        cov = torch.inverse(hessian)
        var_logit = cov[0, 0].item()
        # Delta method for lam = sigmoid(logit_lam)
        # d(sigmoid)/d(logit) = lam * (1 - lam)
        dlam = lam_opt * (1.0 - lam_opt)
        lam_std = float(np.sqrt(max(1e-8, var_logit * (dlam ** 2))))
    except Exception as e:
        lam_std = 0.03

    ci_95 = [float(np.clip(lam_opt - 1.96 * lam_std, 0.0, 1.0)),
             float(np.clip(lam_opt + 1.96 * lam_std, 0.0, 1.0))]

    return dict(
        lam_hat=float(lam_opt),
        lam_std=float(lam_std),
        ci_95=ci_95,
        r_hat=float(r_opt),
        z=z
    )


def run_benchmark():
    print("==================================================================")
    print("RUNNING CANDIDATE METHODS BENCHMARK ON SYNTHETIC TEST CASES")
    print("==================================================================")

    test_cases = [
        dict(N=200, lam=0.8, r=1.2, K=2, c=15.0, seed=42),
        dict(N=300, lam=0.8, r=1.2, K=2, c=15.0, seed=7),
        dict(N=200, lam=0.3, r=1.2, K=2, c=15.0, seed=10),
        dict(N=300, lam=0.5, r=0.8, K=4, c=15.0, seed=123),
        dict(N=250, lam=0.9, r=1.5, K=3, c=12.0, seed=999),
    ]

    for tc in test_cases:
        print(f"\n--- Testing Network: N={tc['N']}, lam_true={tc['lam']}, r_true={tc['r']}, K={tc['K']}, seed={tc['seed']} ---")
        G, data = generate_dual_weighted_network(
            N=tc["N"], lam=tc["lam"], c=tc["c"], r=tc["r"], K=tc["K"], seed=tc["seed"]
        )
        X = data["X"]
        W = data["W"]

        # Run M1: Moment Inversion
        t0 = time.time()
        res_m1 = method_moment_inversion(X, W, c_target=tc["c"], K_est=tc["K"])
        t_m1 = time.time() - t0
        print(f"M1 (Moment Inversion) : lam_hat={res_m1['lam_hat']:.4f} | r_hat={res_m1['r_hat']:.4f} | time={t_m1:.2f}s")

        # Run M3: Profile Likelihood Hessian
        t0 = time.time()
        res_m3 = method_profile_hessian(X, W, c_target=tc["c"], K_est=tc["K"], n_grid=15)
        t_m3 = time.time() - t0
        cov_m3 = res_m3['ci_95'][0] <= tc['lam'] <= res_m3['ci_95'][1]
        print(f"M3 (Profile Likelihood): lam_hat={res_m3['lam_hat']:.4f} +/- {res_m3['lam_std']:.4f} | 95% CI={res_m3['ci_95']} (covered: {cov_m3}) | r_hat={res_m3['r_hat']:.4f} | time={t_m3:.2f}s")

        # Run M4: Cluster-Collapsed MLE
        t0 = time.time()
        res_m4 = method_cluster_collapsed_mle(X, W, c_target=tc["c"], K_est=tc["K"])
        t_m4 = time.time() - t0
        cov_m4 = res_m4['ci_95'][0] <= tc['lam'] <= res_m4['ci_95'][1]
        print(f"M4 (Cluster Collapsed): lam_hat={res_m4['lam_hat']:.4f} +/- {res_m4['lam_std']:.4f} | 95% CI={res_m4['ci_95']} (covered: {cov_m4}) | r_hat={res_m4['r_hat']:.4f} | time={t_m4:.2f}s")


if __name__ == "__main__":
    run_benchmark()
