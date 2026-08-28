import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# BLOCCA i thread interni di PyTorch per processo worker
torch.set_num_threads(1)

def _optimize_worker(task):
    lr, t_adam, net_item, T_LBFGS = task

    N = net_item["N"]
    c_target = net_item["c_target"]
    log_r0 = net_item["log_r0"]

    D_phys_t = torch.from_numpy(net_item["D_phys"]).float()
    W_obs_u  = torch.from_numpy(net_item["W_obs_u"]).float()
    iu_t     = torch.from_numpy(net_item["iu"]).long()
    ju_t     = torch.from_numpy(net_item["ju"]).long()
    eye_mask = torch.from_numpy(net_item["eye_mask"]).float()

    logit_lam = torch.tensor(0.0, requires_grad=True)
    log_r     = torch.tensor(log_r0, requires_grad=True)
    S_param   = torch.from_numpy(net_item["S_init"]).float().requires_grad_(True)

    adam_opt = torch.optim.Adam([
        {"params": [logit_lam], "lr": lr * 2.0},
        {"params": [log_r],     "lr": lr},
        {"params": [S_param],   "lr": lr * 5.0}
    ])

    # 1. Warmup con Adam
    for _ in range(t_adam):
        adam_opt.zero_grad(set_to_none=True)
        lam = torch.sigmoid(logit_lam)
        r = torch.exp(torch.clamp(log_r, log_r0 - 3.0, log_r0 + 3.0))

        K_sp = torch.exp(-D_phys_t / r) * eye_mask
        D_soc = torch.cdist(S_param, S_param)
        K_soc = torch.exp(-D_soc) * eye_mask

        W_raw_t = (1.0 - lam) * K_sp + lam * K_soc
        rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
        W_pred_t = 0.5 * c_target * (W_raw_t / rs + W_raw_t / rs.t())

        wu = W_pred_t[iu_t, ju_t]
        loss = (wu - W_obs_u * torch.log(wu + 1e-12)).sum() + 0.5 * ((log_r - log_r0) / 2.0)**2
        loss.backward()
        adam_opt.step()

    # 2. Rifinitura con L-BFGS
    lbfgs = torch.optim.LBFGS(
        [logit_lam, log_r, S_param], 
        lr=0.5, 
        max_iter=T_LBFGS, 
        history_size=10, 
        line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        lam = torch.sigmoid(logit_lam)
        r = torch.exp(torch.clamp(log_r, log_r0 - 3.0, log_r0 + 3.0))

        K_sp = torch.exp(-D_phys_t / r) * eye_mask
        D_soc = torch.cdist(S_param, S_param)
        K_soc = torch.exp(-D_soc) * eye_mask

        W_raw_t = (1.0 - lam) * K_sp + lam * K_soc
        rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
        W_pred_t = 0.5 * c_target * (W_raw_t / rs + W_raw_t / rs.t())

        wu = W_pred_t[iu_t, ju_t]
        loss = (wu - W_obs_u * torch.log(wu + 1e-12)).sum() + 0.5 * ((log_r - log_r0) / 2.0)**2
        loss.backward()
        return loss

    try:
        lbfgs.step(closure)
        lam_hat = float(torch.sigmoid(logit_lam).item())
        err_lam = abs(lam_hat - net_item["lambda_true"])
        S_hat = S_param.detach().numpy()
        D_soc_hat = cdist(S_hat, S_hat)[net_item["iu"], net_item["ju"]]
        D_soc_gt  = net_item["D_soc_flat"]
        rho_soc, _ = pearsonr(D_soc_hat, D_soc_gt)
        if np.isnan(rho_soc):
            rho_soc = 0.0
    except Exception:
        err_lam = 1.0
        rho_soc = 0.0

    return {"lr": lr, "t_adam": t_adam, "err_lam": err_lam, "rho_soc": rho_soc}
