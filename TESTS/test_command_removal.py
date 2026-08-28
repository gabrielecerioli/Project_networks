"""
Experiment: Test hypothesis - What happens to lambda when removing long command interneurons
and long-range synaptic connections in C. elegans?
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
import torch

device = torch.device("cpu")

def run_experiment():
    G = nx.read_graphml("../data/c_elegans/celegans_connectome.graphml")
    nodes = list(G.nodes())
    N = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    cell_names = [G.nodes[n].get("cell_name", n) for n in nodes]
    
    coords = []
    for n in nodes:
        coords.append([float(G.nodes[n]["x"]), float(G.nodes[n]["y"]), float(G.nodes[n]["z"])])
    X = np.array(coords)

    # Base weight matrix
    W_dir = np.zeros((N, N))
    for u, v, d in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        W_dir[i, j] += float(d.get("weight", 1.0))
    W_full = 0.5 * (W_dir + W_dir.T)
    D_phys = cdist(X, X)

    # Command interneurons list
    command_neurons = {"AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER", "PVCL", "PVCR", "DVA", "AVKL", "AVKR"}
    command_indices = {i for i, name in enumerate(cell_names) if name in command_neurons}
    print(f"Identified {len(command_indices)} command interneurons: {[cell_names[i] for i in sorted(command_indices)]}")

    # Solver helper
    def solve_network(X_sub, W_sub):
        n_sub = len(X_sub)
        c_sub = float(W_sub.sum(axis=1).mean())
        if c_sub < 1e-6:
            return None
        D_p = cdist(X_sub, X_sub)
        np.fill_diagonal(D_p, 1e9)
        iu, ju = np.triu_indices(n_sub, k=1)

        D_phys_t = torch.tensor(D_p, dtype=torch.float32, device=device)
        W_obs_u = torch.tensor(W_sub[iu, ju], dtype=torch.float32, device=device)
        iu_t = torch.tensor(iu, dtype=torch.long, device=device)
        ju_t = torch.tensor(ju, dtype=torch.long, device=device)
        eye_mask = 1.0 - torch.eye(n_sub, device=device)

        r0 = float(2.0 * D_p.min(axis=1).mean())
        log_r0 = float(np.log(r0))

        K_sp0 = np.exp(-D_p / r0)
        np.fill_diagonal(K_sp0, 0.0)
        s_sp0 = K_sp0.sum(axis=1, keepdims=True) + 1e-12
        W_sp0 = 0.5 * c_sub * (K_sp0 / s_sp0 + K_sp0 / s_sp0.T)
        R_init = np.maximum(0.0, W_sub - W_sp0)

        U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
        S_init = U[:, :2] * np.sqrt(sv[:2])
        S_init = (S_init - S_init.mean(axis=0)) / (S_init.std() + 1e-9) * 3.0

        logit_lam = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
        log_r = torch.tensor(log_r0, dtype=torch.float32, requires_grad=True, device=device)
        S_param = torch.tensor(S_init, dtype=torch.float32, requires_grad=True, device=device)

        adam_opt = torch.optim.Adam([
            {"params": [logit_lam], "lr": 0.08},
            {"params": [log_r], "lr": 0.05},
            {"params": [S_param], "lr": 0.10}
        ])
        for _ in range(50):
            adam_opt.zero_grad()
            lam = torch.sigmoid(logit_lam)
            r = torch.exp(torch.clamp(log_r, log_r0 - 3.0, log_r0 + 3.0))
            K_sp = torch.exp(-D_phys_t / r) * eye_mask

            diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
            dist_sq = (diff ** 2).sum(-1) + 1e-12
            D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(n_sub, device=device)
            K_soc = torch.exp(-D_soc / 1.0) * eye_mask

            W_raw = (1.0 - lam) * K_sp + lam * K_soc
            row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
            W_pred = 0.5 * c_sub * (W_raw / row_sums + W_raw / row_sums.t())

            w_pred_u = W_pred[iu_t, ju_t]
            loss = (w_pred_u - W_obs_u * torch.log(w_pred_u + 1e-12)).sum()
            loss = loss + 0.5 * ((log_r - log_r0) / 2.0) ** 2
            loss.backward()
            adam_opt.step()

        lbfgs_opt = torch.optim.LBFGS(
            [logit_lam, log_r, S_param],
            lr=0.5,
            max_iter=45,
            history_size=10,
            line_search_fn="strong_wolfe"
        )
        def closure():
            lbfgs_opt.zero_grad()
            lam = torch.sigmoid(logit_lam)
            r = torch.exp(torch.clamp(log_r, log_r0 - 3.0, log_r0 + 3.0))
            K_sp = torch.exp(-D_phys_t / r) * eye_mask
            diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
            dist_sq = (diff ** 2).sum(-1) + 1e-12
            D_soc = torch.sqrt(dist_sq) + 1e9 * torch.eye(n_sub, device=device)
            K_soc = torch.exp(-D_soc / 1.0) * eye_mask
            W_raw = (1.0 - lam) * K_sp + lam * K_soc
            row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12
            W_pred = 0.5 * c_sub * (W_raw / row_sums + W_raw / row_sums.t())
            w_pred_u = W_pred[iu_t, ju_t]
            loss = (w_pred_u - W_obs_u * torch.log(w_pred_u + 1e-12)).sum()
            loss = loss + 0.5 * ((log_r - log_r0) / 2.0) ** 2
            loss.backward()
            return loss

        lbfgs_opt.step(closure)
        lam_est = float(torch.sigmoid(logit_lam).item())
        r_est = float(torch.exp(torch.clamp(log_r, log_r0 - 3.0, log_r0 + 3.0)).item())
        return lam_est, r_est

    # Case 0: Full connectome
    lam0, r0_est = solve_network(X, W_full)
    print(f"\n[Case 0] Full Connectome: lambda_hat = {lam0:.4f}, r_hat = {r0_est:.1f} um")

    # Case 1: Zero out connections belonging to command interneurons
    W_no_command = W_full.copy()
    for idx in command_indices:
        W_no_command[idx, :] = 0.0
        W_no_command[:, idx] = 0.0
    lam1, r1 = solve_network(X, W_no_command)
    print(f"[Case 1] Removing all Command Interneuron edges: lambda_hat = {lam1:.4f}, r_hat = {r1:.1f} um")

    # Case 2: Complete removal of all long-range edges d > 300 um
    W_no_long300 = W_full.copy()
    W_no_long300[D_phys > 300.0] = 0.0
    lam2, r2 = solve_network(X, W_no_long300)
    print(f"[Case 2] Filter all long-range edges (d > 300 um): lambda_hat = {lam2:.4f}, r_hat = {r2:.1f} um")

    # Case 3: Complete removal of all long-range edges d > 150 um
    W_no_long150 = W_full.copy()
    W_no_long150[D_phys > 150.0] = 0.0
    lam3, r3 = solve_network(X, W_no_long150)
    print(f"[Case 3] Filter all long-range edges (d > 150 um): lambda_hat = {lam3:.4f}, r_hat = {r3:.1f} um")

    # Case 4: Complete removal of all long-range edges d > 75 um (strictly local neighborhood)
    W_no_long75 = W_full.copy()
    W_no_long75[D_phys > 75.0] = 0.0
    lam4, r4 = solve_network(X, W_no_long75)
    print(f"[Case 4] Filter all long-range edges (d > 75 um): lambda_hat = {lam4:.4f}, r_hat = {r4:.1f} um")

    # Case 5: Head neurons only (Nerve ring, x < 150 um)
    head_mask = X[:, 0] < 150.0
    X_head = X[head_mask]
    W_head = W_full[head_mask][:, head_mask]
    lam5, r5 = solve_network(X_head, W_head)
    print(f"[Case 5] Head nerve ring only (N={head_mask.sum()}, x < 150 um): lambda_hat = {lam5:.4f}, r_hat = {r5:.1f} um")

if __name__ == "__main__":
    run_experiment()
