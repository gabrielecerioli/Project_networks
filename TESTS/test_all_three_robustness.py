import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

device = torch.device("cpu")

def sample_synthetic_network(N=None, lambda_true=None, r_true=None, sigma_true=None, d_s=2, seed=None):
    if seed is not None:
        np.random.seed(seed); torch.manual_seed(seed)
    
    if N is None:
        N = int(np.random.randint(100, 301)) # Fast for benchmark testing
    if lambda_true is None:
        lambda_true = float(np.random.uniform(0.05, 0.95))
    if r_true is None:
        r_true = float(np.exp(np.random.uniform(np.log(0.03), np.log(0.50))))
    if sigma_true is None:
        sigma_true = float(np.exp(np.random.uniform(np.log(0.03), np.log(0.50))))
    
    X_phys = np.random.uniform(0.0, 1.0, size=(N, 2))
    
    K = int(np.random.randint(2, 9))
    gmm_means = np.random.uniform(0.1, 0.9, size=(K, d_s))
    gmm_covs = [np.diag(np.random.uniform(0.01, 0.05, size=d_s)) for _ in range(K)]
    weights = np.random.dirichlet(np.ones(K))
    
    comp_choices = np.random.choice(K, size=N, p=weights)
    S_true = np.zeros((N, d_s))
    for k in range(K):
        idx_k = np.where(comp_choices == k)[0]
        if len(idx_k) > 0:
            S_true[idx_k] = np.random.multivariate_normal(gmm_means[k], gmm_covs[k], size=len(idx_k))
    
    S_min, S_max = S_true.min(axis=0), S_true.max(axis=0)
    S_true = (S_true - S_min) / (S_max - S_min + 1e-9)
    
    D_phys = cdist(X_phys, X_phys); np.fill_diagonal(D_phys, 1e9)
    D_soc = cdist(S_true, S_true); np.fill_diagonal(D_soc, 1e9)
    
    K_sp = np.exp(-D_phys / r_true); np.fill_diagonal(K_sp, 0.0)
    K_soc = np.exp(-D_soc / sigma_true); np.fill_diagonal(K_soc, 0.0)
    
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
        "D_phys": D_phys, "D_soc": D_soc, "S_true": S_true, "W_obs": W_obs, "W_pred": W_pred
    }

def fit_model(net, lr_adam=0.01, t_adam=40, t_lbfgs=40, sigma_gauge=1.0):
    N = net["N"]
    W = net["W_obs"]
    D_phys = net["D_phys"]
    iu, ju = np.triu_indices(N, k=1)
    
    c_target = float(W.sum(axis=1).mean())
    r0 = float(2.0 * D_phys.min(axis=1).mean())
    log_r0 = float(np.log(r0))
    
    D_phys_t = torch.tensor(D_phys, dtype=torch.float32, device=device)
    W_obs_u  = torch.tensor(W[iu, ju], dtype=torch.float32, device=device)
    iu_t     = torch.tensor(iu, dtype=torch.long, device=device)
    ju_t     = torch.tensor(ju, dtype=torch.long, device=device)
    eye_mask = 1.0 - torch.eye(N, device=device)
    
    # SVD Init
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
    
    # Phase A: Adam Warmup
    adam_losses = []
    if t_adam > 0:
        adam_opt = torch.optim.Adam([
            {"params": [logit_lam], "lr": lr_adam},
            {"params": [log_r],     "lr": lr_adam},
            {"params": [S_param],   "lr": lr_adam}
        ])
        for _ in range(t_adam):
            adam_opt.zero_grad()
            lam = torch.sigmoid(logit_lam)
            r   = torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3))
            K_sp = torch.exp(-D_phys_t / r) * eye_mask
            diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
            D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(N, device=device)
            K_soc = torch.exp(-D_soc / sigma_gauge) * eye_mask
            W_raw_t = (1-lam)*K_sp + lam*K_soc
            rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
            W_pred_t = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
            wu = W_pred_t[iu_t, ju_t]
            loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
            loss.backward()
            adam_opt.step()
            adam_losses.append(loss.item())
            
    # Phase B: L-BFGS Polish
    grad_norms = []
    lbfgs_losses = []
    if t_lbfgs > 0:
        lbfgs = torch.optim.LBFGS([logit_lam, log_r, S_param], lr=0.5,
                                  max_iter=t_lbfgs, history_size=10, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            lam = torch.sigmoid(logit_lam)
            r   = torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3))
            K_sp = torch.exp(-D_phys_t / r) * eye_mask
            diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
            D_soc = torch.sqrt((diff**2).sum(-1)+1e-12) + 1e9*torch.eye(N, device=device)
            K_soc = torch.exp(-D_soc / sigma_gauge) * eye_mask
            W_raw_t = (1-lam)*K_sp + lam*K_soc
            rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
            W_pred_t = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
            wu = W_pred_t[iu_t, ju_t]
            loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
            loss.backward()
            
            # Record max gradient norm
            g_lam = logit_lam.grad.abs().item() if logit_lam.grad is not None else 0.0
            g_r   = log_r.grad.abs().item() if log_r.grad is not None else 0.0
            g_s   = S_param.grad.abs().max().item() if S_param.grad is not None else 0.0
            grad_norms.append(max(g_lam, g_r, g_s))
            lbfgs_losses.append(loss.item())
            return loss
        lbfgs.step(closure)
        
    lam_hat = float(torch.sigmoid(logit_lam).item())
    r_hat   = float(torch.exp(torch.clamp(log_r, log_r0-3, log_r0+3)).item())
    
    # Fidelity rho_soc
    S_hat = S_param.detach().cpu().numpy()
    D_soc_hat = cdist(S_hat, S_hat)[iu, ju]
    D_soc_gt  = net["D_soc"][iu, ju]
    rho_soc, _ = pearsonr(D_soc_hat, D_soc_gt)
    
    # Final gradient norm
    final_gnorm = grad_norms[-1] if grad_norms else 0.0
    
    return {
        "lam_hat": lam_hat,
        "r_hat": r_hat,
        "err_lam": abs(lam_hat - net["lambda_true"]),
        "err_r_rel": abs(r_hat - net["r_true"]) / net["r_true"],
        "rho_soc": rho_soc,
        "final_gnorm": final_gnorm,
        "adam_losses": adam_losses,
        "lbfgs_losses": lbfgs_losses,
        "grad_norms": grad_norms
    }

print("Running quick test of fit_model...")
test_net = sample_synthetic_network(N=100, seed=10)
res = fit_model(test_net, lr_adam=0.01, t_adam=40, t_lbfgs=40)
print(f"Fit result: |Δλ| = {res['err_lam']:.4f}, |Δr|/r = {res['err_r_rel']:.4f}, ρ_soc = {res['rho_soc']:.4f}, g_norm = {res['final_gnorm']:.2e}")
