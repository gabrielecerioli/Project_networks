import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

device = torch.device("cpu")

def generate_synthetic_network(N=100, lambda_true=0.50, r_true=15.0, sigma_gauge=1.0, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    # Physical 3D coords on a sphere or 2D plane
    X_phys = np.random.uniform(-50, 50, size=(N, 2))
    D_phys = cdist(X_phys, X_phys)
    np.fill_diagonal(D_phys, 1e9)
    
    # Latent social space coords (2D)
    S_true = np.random.randn(N, 2) * 2.0
    diff = S_true[:, None, :] - S_true[None, :, :]
    D_soc = np.sqrt((diff**2).sum(-1) + 1e-12)
    np.fill_diagonal(D_soc, 1e9)
    
    K_sp = np.exp(-D_phys / r_true)
    np.fill_diagonal(K_sp, 0.0)
    K_soc = np.exp(-D_soc / sigma_gauge)
    np.fill_diagonal(K_soc, 0.0)
    
    K_raw = (1.0 - lambda_true) * K_sp + lambda_true * K_soc
    c_target = 10.0
    row_sum = K_raw.sum(axis=1, keepdims=True) + 1e-12
    W_clean = 0.5 * c_target * (K_raw / row_sum + K_raw / row_sum.T)
    
    # Poisson noise
    iu, ju = np.triu_indices(N, k=1)
    W_obs = np.zeros_like(W_clean)
    lam_poiss = W_clean[iu, ju]
    w_sampled = np.random.poisson(lam_poiss).astype(float)
    W_obs[iu, ju] = w_sampled
    W_obs[ju, iu] = w_sampled
    
    return D_phys, W_obs, S_true, lambda_true, r_true

# Test 1: Sweep on Learning Rate eta_Adam
print("Testing Learning Rate Sensitivity...")
lr_list = np.logspace(-4, -1, 10)
results_lr = []

for lr in lr_list:
    errors_lam = []
    rhos_soc = []
    losses = []
    
    for seed in [1, 2, 3]:
        D_phys, W, S_true, lam_true, r_true = generate_synthetic_network(N=80, lambda_true=0.45, r_true=18.0, seed=seed)
        N = len(W)
        iu, ju = np.triu_indices(N, k=1)
        c_target = float(W.sum(axis=1).mean())
        r0 = float(2.0 * D_phys.min(axis=1).mean())
        log_r0 = float(np.log(r0))
        
        D_phys_t = torch.tensor(D_phys, dtype=torch.float32, device=device)
        W_obs_u  = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
        iu_t     = torch.tensor(iu, dtype=torch.long, device=device)
        ju_t     = torch.tensor(ju, dtype=torch.long, device=device)
        eye_mask = 1.0 - torch.eye(N, device=device)
        
        # SVD init
        K_sp0 = np.exp(-D_phys / r0)
        np.fill_diagonal(K_sp0, 0.0)
        s_sp0 = K_sp0.sum(axis=1, keepdims=True) + 1e-12
        W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
        R_init = np.maximum(0.0, W - W_sp0)
        U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
        S_init = U[:, :2] * np.sqrt(sv[:2])
        S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0
        
        logit_lam = torch.tensor(0.0, requires_grad=True, device=device)
        log_r     = torch.tensor(log_r0, requires_grad=True, device=device)
        S_param   = torch.tensor(S_init, requires_grad=True, device=device)
        
        # Adam phase with tested lr
        adam_opt = torch.optim.Adam([
            {"params": [logit_lam], "lr": lr},
            {"params": [log_r],     "lr": lr},
            {"params": [S_param],   "lr": lr}
        ])
        for _ in range(40):
            adam_opt.zero_grad()
            lam = torch.sigmoid(logit_lam)
            r   = torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3))
            K_sp = torch.exp(-D_phys_t / r) * eye_mask
            diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
            D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(N, device=device)
            K_soc = torch.exp(-D_soc / 1.0) * eye_mask
            W_raw_t = (1-lam)*K_sp + lam*K_soc
            rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
            W_pred = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
            wu = W_pred[iu_t, ju_t]
            loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
            loss.backward(); adam_opt.step()
            
        # L-BFGS polish (40 iters)
        lbfgs = torch.optim.LBFGS([logit_lam, log_r, S_param], lr=0.5,
                                  max_iter=40, history_size=10, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            lam = torch.sigmoid(logit_lam)
            r   = torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3))
            K_sp = torch.exp(-D_phys_t / r) * eye_mask
            diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
            D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(N, device=device)
            K_soc = torch.exp(-D_soc / 1.0) * eye_mask
            W_raw_t = (1-lam)*K_sp + lam*K_soc
            rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
            W_pred = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
            wu = W_pred[iu_t, ju_t]
            loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
            loss.backward(); return loss
        lbfgs.step(closure)
        
        lam_hat = float(torch.sigmoid(logit_lam).item())
        errors_lam.append(abs(lam_hat - lam_true))
        losses.append(closure().item())
        
        # Social distance correlation
        S_hat_np = S_param.detach().cpu().numpy()
        D_soc_hat = cdist(S_hat_np, S_hat_np)[iu, ju]
        D_soc_gt = cdist(S_true, S_true)[iu, ju]
        r_corr, _ = pearsonr(D_soc_hat, D_soc_gt)
        rhos_soc.append(r_corr)
        
    results_lr.append({
        "lr": lr,
        "err_lam_mean": np.mean(errors_lam),
        "err_lam_std": np.std(errors_lam),
        "rho_soc_mean": np.mean(rhos_soc),
        "rho_soc_std": np.std(rhos_soc),
        "loss_mean": np.mean(losses)
    })
    print(f"  → lr = {lr:.1e} | |Δλ| = {np.mean(errors_lam):.4f} | ρ_soc = {np.mean(rhos_soc):.4f}")

df_lr = pd.DataFrame(results_lr)
print("\nResults DataFrame for Learning Rate Sensitivity:")
print(df_lr)
