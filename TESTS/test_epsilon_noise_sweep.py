import os, time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

output_dir = "Robustness_tests" if os.path.basename(os.getcwd()) != "Robustness_tests" else "."
os.makedirs(output_dir, exist_ok=True)

# LaTeX-style font formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 9.5,
    "axes.edgecolor": "#1e293b",
    "axes.linewidth": 1.0,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.dpi": 150
})

device = torch.device("cpu")
torch.manual_seed(42); np.random.seed(42)

def sample_synthetic_network(N=None, lambda_true=None, r_true=None, sigma_true=None, d_s=2, seed=None):
    if seed is not None:
        np.random.seed(seed); torch.manual_seed(seed)
    if N is None:
        N = int(np.random.randint(120, 250))
    if lambda_true is None:
        lambda_true = float(np.random.uniform(0.10, 0.90))
    if r_true is None:
        r_true = float(np.exp(np.random.uniform(np.log(0.04), np.log(0.40))))
    if sigma_true is None:
        sigma_true = float(np.exp(np.random.uniform(np.log(0.04), np.log(0.40))))
    
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
    W_pred_clean = 0.5 * (W_raw / row_sum + W_raw / row_sum.T)
    
    return {
        "N": N, "lambda_true": lambda_true, "r_true": r_true, "sigma_true": sigma_true,
        "D_phys": D_phys, "D_soc": D_soc, "S_true": S_true, "W_pred_clean": W_pred_clean
    }

# --- Sweep Parameters ---
epsilons = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
N_BENCHMARK_NETWORKS = 15
ETA_ADAM = 1e-2
T_ADAM = 40
T_LBFGS = 40

tag = f"Nnets{N_BENCHMARK_NETWORKS}_Tadam{T_ADAM}_Tlbfgs{T_LBFGS}"
print(f"Testing Epsilon-Noise Sweep across {len(epsilons)} noise levels on {N_BENCHMARK_NETWORKS} networks...")

ensemble_bases = [sample_synthetic_network(seed=700 + i) for i in range(N_BENCHMARK_NETWORKS)]
records_eps = []

for eps in epsilons:
    t_eps = time.time()
    for net_idx, net in enumerate(ensemble_bases):
        N = net["N"]; D_phys = net["D_phys"]; W_clean = net["W_pred_clean"]
        iu, ju = np.triu_indices(N, k=1)
        
        # Inject uniform background Erdős–Rényi noise
        W_noise = np.ones((N, N)) / (N - 1)
        np.fill_diagonal(W_noise, 0.0)
        
        W_pred_noisy = (1.0 - eps) * W_clean + eps * W_noise
        
        # Poisson realization
        rate = N * W_pred_noisy[iu, ju]
        w_obs_u = np.random.poisson(rate).astype(float)
        W_obs = np.zeros((N, N))
        W_obs[iu, ju] = w_obs_u
        W_obs[ju, iu] = w_obs_u
        
        c_target = float(W_obs.sum(axis=1).mean())
        r0 = float(2.0 * D_phys.min(axis=1).mean()); log_r0 = float(np.log(r0))
        
        D_phys_t = torch.tensor(D_phys, dtype=torch.float32, device=device)
        W_obs_u_t = torch.tensor(w_obs_u, dtype=torch.float32, device=device)
        iu_t, ju_t = torch.tensor(iu, dtype=torch.long, device=device), torch.tensor(ju, dtype=torch.long, device=device)
        eye_mask = 1.0 - torch.eye(N, device=device)
        
        # SVD Residual Initialization
        K_sp0 = np.exp(-D_phys / r0); np.fill_diagonal(K_sp0, 0.0)
        s_sp0 = K_sp0.sum(axis=1, keepdims=True) + 1e-12
        W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
        R_init = np.maximum(0.0, W_obs - W_sp0)
        U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
        S_init = U[:, :2] * np.sqrt(sv[:2])
        S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0
        
        logit_lam = torch.tensor(0.0, requires_grad=True, device=device)
        log_r     = torch.tensor(log_r0, requires_grad=True, device=device)
        S_param   = torch.tensor(S_init, requires_grad=True, device=device)
        
        # Phase A: Adam
        adam_opt = torch.optim.Adam([
            {"params": [logit_lam], "lr": ETA_ADAM * 2.0},
            {"params": [log_r],     "lr": ETA_ADAM},
            {"params": [S_param],   "lr": ETA_ADAM * 5.0}
        ])
        for _ in range(T_ADAM):
            adam_opt.zero_grad()
            lam = torch.sigmoid(logit_lam); r = torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3))
            K_sp = torch.exp(-D_phys_t / r) * eye_mask
            diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
            D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(N, device=device)
            K_soc = torch.exp(-D_soc / 1.0) * eye_mask
            W_raw_t = (1-lam)*K_sp + lam*K_soc
            rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
            W_pred_t = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
            wu = W_pred_t[iu_t, ju_t]
            loss = (wu - W_obs_u_t*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
            loss.backward(); adam_opt.step()
            
        # Phase B: L-BFGS
        lbfgs = torch.optim.LBFGS([logit_lam, log_r, S_param], lr=0.5,
                                  max_iter=T_LBFGS, history_size=10, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            lam = torch.sigmoid(logit_lam); r = torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3))
            K_sp = torch.exp(-D_phys_t / r) * eye_mask
            diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
            D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(N, device=device)
            K_soc = torch.exp(-D_soc / 1.0) * eye_mask
            W_raw_t = (1-lam)*K_sp + lam*K_soc
            rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
            W_pred_t = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
            wu = W_pred_t[iu_t, ju_t]
            loss = (wu - W_obs_u_t*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
            loss.backward(); return loss
        
        try:
            lbfgs.step(closure)
            lam_hat = float(torch.sigmoid(logit_lam).item())
            r_hat   = float(torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3)).item())
            err_lam = abs(lam_hat - net["lambda_true"])
            err_r_rel = abs(r_hat - net["r_true"]) / net["r_true"]
            
            S_hat = S_param.detach().cpu().numpy()
            D_soc_hat = cdist(S_hat, S_hat)[iu, ju]
            D_soc_gt  = net["D_soc"][iu, ju]
            rho_soc, _ = pearsonr(D_soc_hat, D_soc_gt)
        except Exception:
            err_lam = 1.0; err_r_rel = 10.0; rho_soc = 0.0
            
        records_eps.append({
            "network_idx": net_idx, "epsilon": eps, "err_lam": err_lam,
            "err_r_rel": err_r_rel, "rho_soc": rho_soc
        })
    print(f"  → Noise ε = {eps:4.2f} | |Δλ| = {np.mean([r['err_lam'] for r in records_eps if r['epsilon']==eps]):.4f} | ρ_soc = {np.mean([r['rho_soc'] for r in records_eps if r['epsilon']==eps]):.4f} ({time.time()-t_eps:.2f}s)")

df_raw = pd.DataFrame(records_eps)
df_summary = df_raw.groupby("epsilon").agg({
    "err_lam": ["mean", "std"],
    "rho_soc": ["mean", "std"],
    "err_r_rel": ["mean", "std"]
}).reset_index()

print("\nSummary Results of Epsilon-Noise Sweep:")
print(df_summary)
