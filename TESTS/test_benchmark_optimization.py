import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

device = torch.device("cpu")
from test_all_three_robustness import sample_synthetic_network

def run_test_lr():
    # Sweep over learning rates eta in [1e-4, 1e-1]
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    records = []
    
    # Run on 10 diverse synthetic networks
    for net_idx in range(10):
        net = sample_synthetic_network(N=120, seed=100 + net_idx)
        for lr in lrs:
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
                W_pred_t = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
                wu = W_pred_t[iu_t, ju_t]
                loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
                loss.backward(); adam_opt.step()
                
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
                W_pred_t = 0.5*c_target*(W_raw_t/rs + W_raw_t/rs.t())
                wu = W_pred_t[iu_t, ju_t]
                loss = (wu - W_obs_u*torch.log(wu+1e-12)).sum() + 0.5*((log_r-log_r0)/2)**2
                loss.backward(); return loss
            lbfgs.step(closure)
            
            lam_hat = float(torch.sigmoid(logit_lam).item())
            err_lam = abs(lam_hat - net["lambda_true"])
            
            S_hat = S_param.detach().cpu().numpy()
            D_soc_hat = cdist(S_hat, S_hat)[iu, ju]
            D_soc_gt  = net["D_soc"][iu, ju]
            rho_soc, _ = pearsonr(D_soc_hat, D_soc_gt)
            
            records.append({
                "net_idx": net_idx,
                "lr": lr,
                "err_lam": err_lam,
                "rho_soc": rho_soc
            })
            
    df_res = pd.DataFrame(records)
    summary = df_res.groupby("lr")[["err_lam", "rho_soc"]].agg(["mean", "std"])
    print("\nSummary of Learning Rate Sensitivity Sweep:")
    print(summary)

run_test_lr()
