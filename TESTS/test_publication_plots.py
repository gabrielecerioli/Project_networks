import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# Publication Style Setup
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13.5,
    "axes.edgecolor": "#1e293b",
    "axes.linewidth": 1.0,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.dpi": 150
})

device = torch.device("cpu")
torch.manual_seed(42); np.random.seed(42)

def sample_synthetic_network(N=None, lambda_true=None, r_true=None, sigma_true=None, d_s=2):
    if N is None:
        N = int(np.random.randint(100, 301))
    if lambda_true is None:
        lambda_true = float(np.random.uniform(0.05, 0.95))
    if r_true is None:
        r_true = float(np.exp(np.random.uniform(np.log(0.03), np.log(0.50))))
    if sigma_true is None:
        sigma_true = float(np.exp(np.random.uniform(np.log(0.03), np.log(0.50))))
    
    X_phys = np.random.uniform(0.0, 1.0, size=(N, 2))
    
    K = int(np.random.randint(2, 9))
    gmm_means = np.random.uniform(0.1, 0.9, size=(K, d_s))
    gmm_covs  = [np.diag(np.random.uniform(0.01, 0.05, size=d_s)) for _ in range(K)]
    weights   = np.random.dirichlet(np.ones(K))
    comp_choices = np.random.choice(K, size=N, p=weights)
    
    S_true = np.zeros((N, d_s))
    for k in range(K):
        idx_k = np.where(comp_choices == k)[0]
        if len(idx_k) > 0:
            S_true[idx_k] = np.random.multivariate_normal(gmm_means[k], gmm_covs[k], size=len(idx_k))
    S_min, S_max = S_true.min(axis=0), S_true.max(axis=0)
    S_true = (S_true - S_min) / (S_max - S_min + 1e-9)
    
    D_phys = cdist(X_phys, X_phys); np.fill_diagonal(D_phys, 1e9)
    D_soc  = cdist(S_true, S_true); np.fill_diagonal(D_soc, 1e9)
    
    K_sp  = np.exp(-D_phys / r_true);     np.fill_diagonal(K_sp, 0.0)
    K_soc = np.exp(-D_soc / sigma_true);  np.fill_diagonal(K_soc, 0.0)
    
    W_raw = (1.0 - lambda_true) * K_sp + lambda_true * K_soc
    row_sum = W_raw.sum(axis=1, keepdims=True) + 1e-12
    W_pred = 0.5 * (W_raw / row_sum + W_raw / row_sum.T)
    
    iu, ju = np.triu_indices(N, k=1)
    rate = N * W_pred[iu, ju]
    w_obs_u = np.random.poisson(rate).astype(float)
    
    W_obs = np.zeros((N, N))
    W_obs[iu, ju] = w_obs_u
    W_obs[ju, iu] = w_obs_u
    
    return {
        "N": N, "lambda_true": lambda_true, "r_true": r_true, "sigma_true": sigma_true,
        "D_phys": D_phys, "D_soc": D_soc, "S_true": S_true, "W_obs": W_obs
    }

print("Running test of publication plot generation...")
# Test generating sample dataset
ensemble = [sample_synthetic_network() for _ in range(5)]
print(f"Generated {len(ensemble)} test networks.")
