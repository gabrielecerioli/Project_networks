import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# Limita PyTorch a 1 thread per processo worker
torch.set_num_threads(1)

def _eps_noise_worker(task):
    net_idx, eps, seed_poisson, net_item, ETA_ADAM, T_ADAM, T_LBFGS = task

    N = net_item["N"]
    D_phys = net_item["D_phys"]
    W_clean = net_item["W_pred_clean"]
    iu = net_item["iu"]
    ju = net_item["ju"]
    eye_mask = net_item["eye_mask"]

    # Generazione rumore e realizzazione stocastica di Poisson
    rng = np.random.RandomState(seed_poisson)
    W_noise = np.ones((N, N), dtype=np.float32) / (N - 1)
    np.fill_diagonal(W_noise, 0.0)
    W_pred_noisy = (1.0 - eps) * W_clean + eps * W_noise

    rate = N * W_pred_noisy[iu, ju]
    w_obs_u = rng.poisson(rate).astype(np.float32)

    W_obs = np.zeros((N, N), dtype=np.float32)
    W_obs[iu, ju] = w_obs_u
    W_obs[ju, iu] = w_obs_u

    c_target = float(W_obs.sum(axis=1).mean())
    r0 = float(2.0 * D_phys.min(axis=1).mean())
    log_r0 = float(np.log(r0))

    # Inizializzazione SVD sul residuo
    K_sp0 = np.exp(-D_phys / r0)
    np.fill_diagonal(K_sp0, 0.0)
    s_sp0 = K_sp0.sum(axis=1, keepdims=True) + 1e-12
    W_sp0 = 0.5 * c_target * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
    R_init = np.maximum(0.0, W_obs - W_sp0)
    U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
    S_init = U[:, :2] * np.sqrt(sv[:2])
    S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0

    # Allocazione tensori PyTorch locali
    D_phys_t   = torch.from_numpy(D_phys).float()
    W_obs_u_t  = torch.from_numpy(w_obs_u).float()
    iu_t       = torch.from_numpy(iu).long()
    ju_t       = torch.from_numpy(ju).long()
    eye_mask_t = torch.from_numpy(eye_mask).float()

    logit_lam = torch.tensor(0.0, requires_grad=True)
    log_r     = torch.tensor(log_r0, requires_grad=True)
    S_param   = torch.from_numpy(S_init.astype(np.float32)).requires_grad_(True)

    # Fase A: Adam Warmup
    adam_opt = torch.optim.Adam([
        {"params": [logit_lam], "lr": ETA_ADAM * 2.0},
        {"params": [log_r],     "lr": ETA_ADAM},
        {"params": [S_param],   "lr": ETA_ADAM * 5.0}
    ])

    for _ in range(T_ADAM):
        adam_opt.zero_grad(set_to_none=True)
        lam = torch.sigmoid(logit_lam)
        r = torch.exp(torch.clamp(log_r, log_r0 - 3.0, log_r0 + 3.0))

        K_sp = torch.exp(-D_phys_t / r) * eye_mask_t
        D_soc = torch.cdist(S_param, S_param)
        K_soc = torch.exp(-D_soc) * eye_mask_t

        W_raw_t = (1.0 - lam) * K_sp + lam * K_soc
        rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
        W_pred_t = 0.5 * c_target * (W_raw_t / rs + W_raw_t / rs.t())

        wu = W_pred_t[iu_t, ju_t]
        loss = (wu - W_obs_u_t * torch.log(wu + 1e-12)).sum() + 0.5 * ((log_r - log_r0) / 2.0)**2
        loss.backward()
        adam_opt.step()

    # Fase B: L-BFGS Refinement
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

        K_sp = torch.exp(-D_phys_t / r) * eye_mask_t
        D_soc = torch.cdist(S_param, S_param)
        K_soc = torch.exp(-D_soc) * eye_mask_t

        W_raw_t = (1.0 - lam) * K_sp + lam * K_soc
        rs = W_raw_t.sum(dim=1, keepdim=True) + 1e-12
        W_pred_t = 0.5 * c_target * (W_raw_t / rs + W_raw_t / rs.t())

        wu = W_pred_t[iu_t, ju_t]
        loss = (wu - W_obs_u_t * torch.log(wu + 1e-12)).sum() + 0.5 * ((log_r - log_r0) / 2.0)**2
        loss.backward()
        return loss

    try:
        lbfgs.step(closure)
        lam_hat = float(torch.sigmoid(logit_lam).item())
        r_hat   = float(torch.exp(torch.clamp(log_r, log_r0 - 3.0, log_r0 + 3.0)).item())
        err_lam = abs(lam_hat - net_item["lambda_true"])
        err_r_rel = abs(r_hat - net_item["r_true"]) / net_item["r_true"]

        S_hat = S_param.detach().numpy()
        D_soc_hat = cdist(S_hat, S_hat)[iu, ju]
        D_soc_gt  = net_item["D_soc_flat"]
        rho_soc, _ = pearsonr(D_soc_hat, D_soc_gt)
        if np.isnan(rho_soc):
            rho_soc = 0.0
    except Exception:
        err_lam = 1.0
        err_r_rel = 10.0
        rho_soc = 0.0

    return {
        "network_idx": net_idx,
        "epsilon": eps,
        "err_lam": err_lam,
        "err_r_rel": err_r_rel,
        "rho_soc": rho_soc
    }
