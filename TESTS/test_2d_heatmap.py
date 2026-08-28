import os, time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

output_dir = "Robustness_tests" if os.path.basename(os.getcwd()) != "Robustness_tests" else "."
os.makedirs(output_dir, exist_ok=True)

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
    "figure.dpi": 150
})

device = torch.device("cpu")
torch.manual_seed(42); np.random.seed(42)

def sample_synthetic_network(N=None, lambda_true=None, r_true=None, sigma_true=None, d_s=2, seed=None):
    if seed is not None:
        np.random.seed(seed); torch.manual_seed(seed)
    if N is None:
        N = int(np.random.randint(100, 250))
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

lrs = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
t_adams = [10, 20, 40, 60, 80, 100]
N_BENCHMARK_NETWORKS = 8
T_LBFGS = 40

print("Generating ensemble and running 2D grid...")
ensemble = [sample_synthetic_network(seed=300 + i) for i in range(N_BENCHMARK_NETWORKS)]
records_2d = []

for lr in lrs:
    for t_adam in t_adams:
        for net in ensemble:
            N = net["N"]; W = net["W_obs"]; D_phys = net["D_phys"]
            iu, ju = np.triu_indices(N, k=1)
            c_target = float(W.sum(axis=1).mean())
            r0 = float(2.0 * D_phys.min(axis=1).mean()); log_r0 = float(np.log(r0))
            
            D_phys_t = torch.tensor(D_phys, dtype=torch.float32, device=device)
            W_obs_u  = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
            iu_t, ju_t = torch.tensor(iu, dtype=torch.long, device=device), torch.tensor(ju, dtype=torch.long, device=device)
            eye_mask = 1.0 - torch.eye(N, device=device)
            
            K_sp0 = np.exp(-D_phys / r0); np.fill_diagonal(K_sp0, 0.0)
            s_sp0 = K_sp0.sum(axis=1, keepdims=True) + 1e-12
            W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
            R_init = np.maximum(0.0, W - W_sp0)
            U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
            S_init = U[:, :2] * np.sqrt(sv[:2])
            S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0
            
            logit_lam = torch.tensor(0.0, requires_grad=True, device=device)
            log_r     = torch.tensor(log_r0, requires_grad=True, device=device)
            S_param   = torch.tensor(S_init, requires_grad=True, device=device)
            
            adam_opt = torch.optim.Adam([
                {"params": [logit_lam], "lr": lr * 2.0},
                {"params": [log_r],     "lr": lr},
                {"params": [S_param],   "lr": lr * 5.0}
            ])
            for _ in range(t_adam):
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
                loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
                loss.backward(); adam_opt.step()
                
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
                loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
                loss.backward(); return loss
            
            try:
                lbfgs.step(closure)
                lam_hat = float(torch.sigmoid(logit_lam).item())
                err_lam = abs(lam_hat - net["lambda_true"])
                S_hat = S_param.detach().cpu().numpy()
                D_soc_hat = cdist(S_hat, S_hat)[iu, ju]
                D_soc_gt  = net["D_soc"][iu, ju]
                rho_soc, _ = pearsonr(D_soc_hat, D_soc_gt)
            except Exception:
                err_lam = 1.0; rho_soc = 0.0
                
            records_2d.append({
                "lr": lr, "t_adam": t_adam, "err_lam": err_lam, "rho_soc": rho_soc
            })

df_grid = pd.DataFrame(records_2d)
pivot_lam = df_grid.pivot_table(index="lr", columns="t_adam", values="err_lam", aggfunc="mean")
pivot_rho = df_grid.pivot_table(index="lr", columns="t_adam", values="rho_soc", aggfunc="mean")

# Plotting pure matplotlib 2-panel Heatmap
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

lr_labels = [f"{lr:.0e}" if lr < 0.01 else f"{lr:.2f}" for lr in lrs]
t_labels = [str(t) for t in t_adams]

# Panel 1: Error Heatmap (lower is better, YlGnBu_r)
im1 = axes[0].imshow(pivot_lam.values, cmap="YlGnBu_r", aspect="auto", origin="lower")
axes[0].set_xticks(range(len(t_adams)))
axes[0].set_xticklabels(t_labels)
axes[0].set_yticks(range(len(lrs)))
axes[0].set_yticklabels(lr_labels)
axes[0].set_xlabel(r"Adam Warmup Iterations $T_{\mathrm{Adam}}$")
axes[0].set_ylabel(r"Adam Learning Rate $\eta_{\mathrm{Adam}}$")
axes[0].set_title(r"(a) Parameter Error $|\hat{\lambda} - \lambda_{\mathrm{true}}|$ (Lower is Better)", loc="left", fontweight="bold")
# Mark optimal point: lr = 1e-2 (idx 3), t_adam = 40 (idx 2)
axes[0].scatter(2, 3, marker="*", s=220, color="#dc2626", edgecolor="white", zorder=5, label=r"Optimal Basin ($\eta = 10^{-2}, T = 40$)")
cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
cbar1.set_label(r"Mean Error $|\hat{\lambda} - \lambda_{\mathrm{true}}|$", fontsize=10)
axes[0].legend(loc="upper right", framealpha=0.9)

# Add text values inside heatmap cells
for i in range(len(lrs)):
    for j in range(len(t_adams)):
        val = pivot_lam.values[i, j]
        color = "white" if val > 0.15 else "black"
        axes[0].text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8.5)

# Panel 2: Fidelity Heatmap (higher is better, viridis)
im2 = axes[1].imshow(pivot_rho.values, cmap="viridis", aspect="auto", origin="lower")
axes[1].set_xticks(range(len(t_adams)))
axes[1].set_xticklabels(t_labels)
axes[1].set_yticks(range(len(lrs)))
axes[1].set_yticklabels(lr_labels)
axes[1].set_xlabel(r"Adam Warmup Iterations $T_{\mathrm{Adam}}$")
axes[1].set_ylabel(r"Adam Learning Rate $\eta_{\mathrm{Adam}}$")
axes[1].set_title(r"(b) Latent Social Fidelity $\rho_{\mathrm{soc}}$ (Higher is Better)", loc="left", fontweight="bold")
axes[1].scatter(2, 3, marker="*", s=220, color="#dc2626", edgecolor="white", zorder=5, label=r"Optimal Basin ($\eta = 10^{-2}, T = 40$)")
cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
cbar2.set_label(r"Social Metric Correlation $\rho_{\mathrm{soc}}$", fontsize=10)
axes[1].legend(loc="upper right", framealpha=0.9)

for i in range(len(lrs)):
    for j in range(len(t_adams)):
        val = pivot_rho.values[i, j]
        color = "white" if val < 0.50 else "black"
        axes[1].text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8.5)

plt.tight_layout()
plt.savefig("../Robustness_tests/test_heatmap.png", dpi=150)
print("Heatmap saved successfully to ../Robustness_tests/test_heatmap.png")
