"""
Diagnostic Investigation of Dual Spatial-Social Network Parameter Inference.
Analyzes:
1. Spatial vs Social weight decomposition
2. Behavior of edge weights vs physical distance D_phys
3. Properties of residual matrices R(lambda, r) = W - W_sp(lambda, r)
4. Spectral properties of social clustering
5. Exact likelihood vs profile ELBO behavior
"""

import sys
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def generate_dual_weighted_network(
    N=800,
    lam=0.5,
    c=15.0,
    r=1.2,
    sigma=1.2,
    K=8,
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

    return G, dict(X=X, S=S, cluster_id=cluster_id, D_phys=D_phys, D_soc=D_soc, W=W, W_raw=W_raw)


def run_diagnostics():
    print("--- Running Diagnostic Analysis ---")
    N = 300
    lam_true = 0.8
    r_true = 1.2
    c_true = 15.0
    sigma_true = 1.2
    K_true = 2
    seed = 7

    G, data = generate_dual_weighted_network(
        N=N, lam=lam_true, c=c_true, r=r_true, sigma=sigma_true,
        K=K_true, L=12.0, L_s=12.0, d_s=2, cluster_std=0.9, seed=seed
    )

    X = data["X"]
    S = data["S"]
    clusters = data["cluster_id"]
    W = data["W"]
    D_phys = data["D_phys"]
    D_soc = data["D_soc"]

    iu, ju = np.triu_indices(N, k=1)
    weights = W[iu, ju]
    d_phys_u = D_phys[iu, ju]
    same_cluster = (clusters[iu] == clusters[ju])

    print(f"Network generated: N={N}, lam={lam_true}, r={r_true}, c={c_true}, K={K_true}")
    print(f"Total edges: {len(weights)}, Mean weight: {weights.mean():.6f}, Total weight per node: {W.sum(axis=1).mean():.4f}")

    # 1. Edge weight vs Physical distance split by same vs different cluster
    mean_w_same = weights[same_cluster].mean()
    mean_w_diff = weights[~same_cluster].mean()
    print(f"Mean weight (same cluster): {mean_w_same:.6f} | Mean weight (diff cluster): {mean_w_diff:.6f}")
    print(f"Ratio same/diff cluster: {mean_w_same / (mean_w_diff + 1e-12):.2f}")

    # Long range behavior: D_phys > 4 * r_true (where spatial component exp(-D/r) < exp(-4) = 0.018)
    long_mask = d_phys_u > 4 * r_true
    print(f"\nLong-range pairs (D_phys > {4*r_true:.1f}): {long_mask.sum()} pairs ({100*long_mask.mean():.1f}%)")
    print(f"  Long-range same cluster mean weight: {weights[long_mask & same_cluster].mean():.6f}")
    print(f"  Long-range diff cluster mean weight: {weights[long_mask & (~same_cluster)].mean():.6f}")
    print(f"  Long-range same cluster min/max: {weights[long_mask & same_cluster].min():.6f} / {weights[long_mask & same_cluster].max():.6f}")
    print(f"  Long-range diff cluster min/max: {weights[long_mask & (~same_cluster)].min():.6f} / {weights[long_mask & (~same_cluster)].max():.6f}")

    # 2. Spectral analysis of W and residual
    # What does W look like after subtracting pure spatial component?
    print("\n--- Spectral Analysis of Raw and Residual Matrices ---")
    for test_r in [0.8, 1.2, 1.6]:
        K_sp = np.exp(-D_phys / test_r)
        np.fill_diagonal(K_sp, 0.0)
        # Symmetrized normalized spatial matrix
        s_sp = K_sp.sum(axis=1, keepdims=True)
        W_sp_norm = 0.5 * c_true * (K_sp / s_sp + K_sp / s_sp.T)

        for test_lam in [0.0, 0.5, 0.8, 1.0]:
            # Residual if we assume mixing (1-test_lam)
            R = W - (1.0 - test_lam) * W_sp_norm
            vals, _ = np.linalg.eigh(R)
            vals = vals[::-1]
            print(f"r={test_r:.2f}, lam={test_lam:.2f} -> Top 5 eigenvalues of R: {vals[:5].round(3)}, bottom 2: {vals[-2:].round(3)}")

if __name__ == "__main__":
    run_diagnostics()
