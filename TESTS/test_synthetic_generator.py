import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.mixture import GaussianMixture

device = torch.device("cpu")

def sample_synthetic_network(N=None, lambda_true=None, r_true=None, sigma_true=None, d_s=2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Parameter Sampling
    if N is None:
        N = int(np.random.randint(100, 2001))
    if lambda_true is None:
        lambda_true = float(np.random.uniform(0.01, 0.99))
    if r_true is None:
        r_true = float(np.exp(np.random.uniform(np.log(0.03), np.log(0.50))))
    if sigma_true is None:
        sigma_true = float(np.exp(np.random.uniform(np.log(0.03), np.log(0.50))))
    
    # 2. Coordinates Sampling
    X_phys = np.random.uniform(0.0, 1.0, size=(N, 2))
    
    # GMM for latent social space with K in [2, 8]
    K = int(np.random.randint(2, 9))
    gmm_means = np.random.uniform(0.1, 0.9, size=(K, d_s))
    gmm_covs = [np.diag(np.random.uniform(0.01, 0.05, size=d_s)) for _ in range(K)]
    weights = np.random.dirichlet(np.ones(K))
    
    # Sample component assignments
    comp_choices = np.random.choice(K, size=N, p=weights)
    S_true = np.zeros((N, d_s))
    for k in range(K):
        idx_k = np.where(comp_choices == k)[0]
        if len(idx_k) > 0:
            S_true[idx_k] = np.random.multivariate_normal(gmm_means[k], gmm_covs[k], size=len(idx_k))
    
    # Min-max scale S_true into [0, 1]^d_s
    S_min, S_max = S_true.min(axis=0), S_true.max(axis=0)
    S_true = (S_true - S_min) / (S_max - S_min + 1e-9)
    
    # 3. Distance Matrices
    D_phys = cdist(X_phys, X_phys)
    np.fill_diagonal(D_phys, 1e9)
    
    D_soc = cdist(S_true, S_true)
    np.fill_diagonal(D_soc, 1e9)
    
    # 4. Bimodal Affinity Kernel
    K_sp = np.exp(-D_phys / r_true)
    np.fill_diagonal(K_sp, 0.0)
    K_soc = np.exp(-D_soc / sigma_true)
    np.fill_diagonal(K_soc, 0.0)
    
    W_raw = (1.0 - lambda_true) * K_sp + lambda_true * K_soc
    
    # 5. Balanced Normalization
    row_sum = W_raw.sum(axis=1, keepdims=True) + 1e-12
    W_pred = 0.5 * (W_raw / row_sum + W_raw / row_sum.T)
    
    # 6. Poisson Sampling
    iu, ju = np.triu_indices(N, k=1)
    rate = N * W_pred[iu, ju]
    w_obs_u = np.random.poisson(rate).astype(float)
    
    W_obs = np.zeros((N, N))
    W_obs[iu, ju] = w_obs_u
    W_obs[ju, iu] = w_obs_u
    
    return {
        "N": N, "lambda_true": lambda_true, "r_true": r_true, "sigma_true": sigma_true,
        "D_phys": D_phys, "D_soc": D_soc, "S_true": S_true, "W_obs": W_obs, "W_pred": W_pred
    }

print("Testing synthetic network generator...")
net = sample_synthetic_network(N=150, seed=42)
print(f"Generated network: N={net['N']}, lambda={net['lambda_true']:.3f}, r={net['r_true']:.3f}, sigma={net['sigma_true']:.3f}")
print(f"Observed matrix shape: {net['W_obs'].shape}, active links: {(net['W_obs'] > 0).sum() // 2}")
