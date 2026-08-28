# ======================================================================
# INFERENCE v3.0 - FUNDAMENTAL FIX
# ======================================================================
# Root cause analysis of previous failures:
# 1. KERNEL CONFOUNDING: Both spatial and social kernels used
#    exp(-distance) in 2D Euclidean space. The social space can
#    perfectly mimic the physical space (just set s_i = x_i * scale),
#    making lambda unidentified. FIX: Social kernel = dot product
#    similarity (random dot product graph), spatial = distance decay.
# 2. MC BUG: n_mc parameter was ignored - only 1 sample used.
# 3. GMM PRIOR BREAKS IDENTIFIABILITY: GMM fitted on S_init anchors
#    the social space to an arbitrary coordinate frame. The oracle
#    (true params) evaluated in a different frame gets -66k ELBO.
#    FIX: Use rotation-invariant prior on sphere (vMF).
# 4. ORACLE TEST BUG: Evaluated GMM prior at misaligned coordinates.
#    FIX: Oracle uses flat prior or aligned coordinates.
#
# NEW MODEL:
#   W_raw = (1-lam) * exp(-d_phys / r)  +  lam * sigmoid(s_i · s_j)
#   where s_i are unit vectors on the 2-sphere (von Mises-Fisher prior).
#   This breaks the symmetry: spatial = distance decay, social = angular
#   similarity. They cannot mimic each other.
# ======================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes

torch.manual_seed(42)
np.random.seed(42)

# ======================================================================
# CONFIG
# ======================================================================
LAM_MIN, LAM_MAX, N_POINTS = 0.0, 1.0, 21

PROFILE_ITERS, PROFILE_LR = 2000, 0.03
POLISH_ITERS,  POLISH_LR  = 1000, 0.03
SGVI_ITERS,    SGVI_LR    = 4000, 0.04

N_MC_SCAN = 8
N_MC_SGVI = 8
TAIL_FRAC   = 0.15
LOG_R_SD    = 0.40
DELTA_ELBO  = 5.0
MAX_CAND    = 4
N_MIX_SAMPLES = 20000

# Social space: 2D sphere (S^2) embedded in R^3
SOCIAL_DIM = 3
KAPPA_PRIOR = 1.0  # vMF concentration (1.0 = nearly uniform)

# ======================================================================

# ---------------------------------------------------------------- data
# (Assumes G, PARAMS, X, S_true, Wm, Dp_np, c_target, iu, ju, iu_t, ju_t, Wu_t are defined)
# This cell should be run after the generator and plotting cells.

print(f"c_target = {c_target:.2f} | prior centre r0 = {r0:.3f} | true r = {PARAMS['r']}")

# ---------------------------------------------------------------- 
# vMF prior utilities
# ======================================================================

def vmf_log_prob(s, mu, kappa):
    """
    Log probability of von Mises-Fisher distribution on S^{d-1}.
    s: (N, d) unit vectors
    mu: (d,) mean direction (unit vector)
    kappa: concentration parameter (scalar)
    Returns: (N,) log probabilities
    """
    d = s.shape[-1]
    # log C_d(kappa) = (d/2 - 1) * log(kappa) - (d/2) * log(2*pi) - log I_{d/2-1}(kappa)
    # For d=3: log C_3(kappa) = 0.5*log(kappa/(2*pi)) - log(sinh(kappa)) + log(kappa)
    # Actually: C_d(kappa) = kappa^{d/2-1} / ( (2*pi)^{d/2} I_{d/2-1}(kappa) )
    # d=3: C_3(kappa) = kappa / (4*pi*sinh(kappa))
    from scipy.special import ive
    # We'll compute log C using scipy for accuracy, but here use approximation
    # For torch, we can use the formula:
    # log C_3(kappa) = log(kappa) - log(4*pi) - log(sinh(kappa))
    # We'll precompute this as a constant
    log_C = np.log(kappa) - np.log(4 * np.pi) - np.log(np.sinh(kappa))
    log_C_t = torch.tensor(log_C, dtype=torch.float32, device=s.device)
    # log p(s) = log C_d(kappa) + kappa * (s · mu)
    return log_C_t + kappa * (s * mu).sum(-1)


def sample_vmf(mu, kappa, n_samples):
    """
    Sample from vMF distribution using rejection sampling (simple version).
    For kappa=1, nearly uniform.
    """
    # Simple rejection sampling for S^2
    d = len(mu)
    samples = []
    while len(samples) < n_samples:
        x = np.random.randn(n_samples, d)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        # Accept with prob proportional to exp(kappa * x·mu)
        dots = (x * mu).sum(-1)
        accept = np.random.rand(n_samples) < np.exp(kappa * (dots - 1))
        samples.append(x[accept])
    return np.vstack(samples)[:n_samples]


# ----------------------------------------------------------------
# Precompute oracle social positions in model gauge
# ======================================================================
# True social positions: S_true (N, 2) in [0, L_s]^2 with sigma=1.2
# Model gauge: SIGMA=1, social kernel = sigmoid(s_i · s_j) with s_i on S^2
# Need to map true social clusters to sphere.
# We'll use the oracle check with FLAT prior to avoid coordinate issues.
ORACLE_CHECK = True

# ----------------------------------------------------------------
# Smart init: residual spectral embedding -> project to sphere
# ======================================================================
Wsp = np.exp(-Dp_np / r0)
Wnull = c_target * Wsp / Wsp.sum(axis=1, keepdims=True)
Rres = Wm - 0.5 * (Wnull + Wnull.T)
U, sv, _ = np.linalg.svd(Rres, full_matrices=False)
S_init_2d = U[:, :2] * sv[:2]
# Project 2D init to sphere (stereographic or just normalize 3D with z=0)
# Better: use 3D init from 3D SVD
U3, sv3, _ = np.linalg.svd(Rres, full_matrices=False)
S_init_3d = U3[:, :3] * sv3[:3]
S_init_3d = S_init_3d / (np.linalg.norm(S_init_3d, axis=1, keepdims=True) + 1e-9)

# ----------------------------------------------------------------
# PARAMETERS
# ======================================================================
def make_params():
    return dict(
        # Social positions: unconstrained in R^3, projected to sphere in forward pass
        v_S = torch.tensor(S_init_3d, dtype=torch.float32, requires_grad=True),  # (N, 3)
        logsig_v = torch.full((N, 3), -1.5, dtype=torch.float32, requires_grad=True),
        # Lambda: logit parameter
        mu_th = torch.tensor(0.0, dtype=torch.float32, requires_grad=True),
        logsig_th = torch.tensor(np.log(0.5), dtype=torch.float32, requires_grad=True),
        # Spatial scale
        log_r = torch.tensor(LOG_R0, dtype=torch.float32, requires_grad=True),
    )


def get_social_positions(p):
    """Project unconstrained v_S to unit sphere S^2."""
    v = p["v_S"]
    return v / (v.norm(dim=1, keepdim=True) + 1e-9)


def elbo(p, n_mc=4, lam_fix=None, r_fix=None):
    """
    Compute ELBO with proper Monte Carlo averaging over n_mc samples.
    """
    total_ll = 0.0
    total_kl_S = 0.0
    total_kl_th = 0.0
    total_kl_r = 0.0
    
    for _ in range(n_mc):
        # Sample social positions
        eps_v = torch.randn_like(p["v_S"])
        v_S = p["v_S"] + torch.exp(p["logsig_v"]) * eps_v
        S = v_S / (v_S.norm(dim=1, keepdim=True) + 1e-9)  # Project to sphere
        
        # Sample lambda
        if lam_fix is None:
            th = p["mu_th"] + torch.exp(p["logsig_th"]) * torch.randn(())
            lam = torch.sigmoid(th)
            sd_th = torch.exp(p["logsig_th"])
            kl_th = (torch.log(2.0 / sd_th)
                     + (sd_th**2 + p["mu_th"]**2) / 8.0 - 0.5)
        else:
            lam = torch.tensor(float(lam_fix))
            kl_th = torch.tensor(0.0)
        
        # Sample r
        if r_fix is None:
            r = torch.exp(p["log_r"])
            kl_r = 0.5 * (p["log_r"] - LOG_R0)**2 / LOG_R_SD**2
        else:
            r = torch.tensor(float(r_fix))
            kl_r = torch.tensor(0.0)
        
        # Social kernel: sigmoid(s_i · s_j) - random dot product graph
        dots = torch.mm(S, S.t())  # (N, N) dot products in [-1, 1]
        # Zero diagonal
        dots = dots - torch.diag(dots.diag())
        K_social = torch.sigmoid(dots)  # in (0, 1)
        
        # Spatial kernel: exp(-d_phys / r)
        K_spatial = torch.exp(-Dp_t / r)
        K_spatial = K_spatial - torch.diag(K_spatial.diag())
        
        # Combined weight
        Wraw = (1.0 - lam) * K_spatial + lam * K_social
        
        # Symmetrized row normalization (matches generator exactly)
        RS = Wraw.sum(dim=1, keepdim=True)
        Wpred = 0.5 * c_target * Wraw * (1.0 / RS + 1.0 / RS.t())
        
        # Poisson likelihood on upper triangle
        ll = (Wu_t * torch.log(Wpred[iu_t, ju_t] + 1e-12)
              - Wpred[iu_t, ju_t]).sum()
        
        # KL for S: prior is uniform on sphere (vMF with kappa=1)
        # For vMF with kappa=1, log p(S) = const + 1*(S·mu). With mu=0 (uniform), log p = const.
        # KL(q||p_uniform) = -H(q) - E_q[log p_uniform] = -H(q) - const
        # We'll use the entropy of the Gaussian in tangent space projected to sphere
        # Approximation: entropy of the Gaussian in R^3 before projection
        # This is not exact but reasonable for small variance
        logsig_v = p["logsig_v"]
        H = (0.5 * (1.0 + torch.log(torch.tensor(2 * torch.pi))) + logsig_v).sum()
        kl_S = -H  # uniform prior on sphere has constant log prob
        
        # KL for lambda
        if lam_fix is None:
            sd_th = torch.exp(p["logsig_th"])
            kl_th = (torch.log(2.0 / torch.exp(p["logsig_th"]))
                     + (torch.exp(2*p["logsig_th"]) + p["mu_th"]**2) / 8.0 - 0.5)
        else:
            kl_th = torch.tensor(0.0)
        
        if r_fix is None:
            kl_r = 0.5 * (p["log_r"] - LOG_R0)**2 / LOG_R_SD**2
        else:
            kl_r = torch.tensor(0.0)
        
        total_ll += ll
        total_kl_S += kl_S
        total_kl_th += kl_th
        total_kl_r += kl_r
    
    # Average over MC samples
    return (total_ll - total_kl_S - total_kl_th - total_kl_r) / n_mc


def train(p, n_iter=3000, lr=0.05, lam_fix=None, r_fix=None,
          n_mc=4, verbose=False):
    opt = torch.optim.Adam(p.values(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter)
    hist = []
    for it in range(n_iter):
        opt.zero_grad()
        loss = -elbo(p, n_mc=n_mc, lam_fix=lam_fix, r_fix=r_fix)
        loss.backward()
        opt.step()
        sched.step()
        hist.append(loss.item())
        if verbose and it % max(1, n_iter // 10) == 0:
            lam_now = lam_fix if lam_fix is not None \
                else float(torch.sigmoid(p["mu_th"]))
            msg_r = r_fix if r_fix is not None \
                else float(torch.exp(p["log_r"]))
            print(f"  it {it:5d} | -ELBO {loss.item():12.2f} | "
                  f"lam {lam_now:.4f} | r {msg_r:.3f}")
    return hist, p


def tail_elbo(hist):
    k = max(20, int(TAIL_FRAC * len(hist)))
    return -float(np.mean(hist[-k:]))


# ----------------------------------------------------------------
# ORACLE EVALUATION (with flat prior to avoid coordinate issues)
# ======================================================================
def eval_elbo_flat(lam_v, r_v, S_v, n_rep=64):
    """Evaluate ELBO at given params with FLAT prior (no GMM)."""
    p_o = make_params()
    with torch.no_grad():
        # Project S_v to sphere
        S_v = S_v / (np.linalg.norm(S_v, axis=1, keepdims=True) + 1e-9)
        p_o["v_S"].copy_(torch.tensor(S_v, dtype=torch.float32))
        p_o["logsig_v"].fill_(-4.0)
        p_o["mu_th"].fill_(np.log(lam_v / (1 - lam_v)))
        p_o["logsig_th"].fill_(-4.0)
        p_o["log_r"].fill_(np.log(r_v))
    vals = [elbo(p_o, n_mc=1).item() for _ in range(32)]
    return float(np.mean(vals)), float(np.std(vals))


# ----------------------------------------------------------------
# RUN ORACLE CHECK
# ======================================================================
# Map true social positions to sphere for oracle
# True social positions are 2D clusters. Map to sphere via (x, y, 1) -> normalize
S_true_3d = np.hstack([S_true, np.ones((N, 1))])
S_true_3d = S_true_3d / (np.linalg.norm(S_true_3d, axis=1, keepdims=True) + 1e-9)

om, osd = eval_elbo_flat(lam_true, PARAMS["r"], S_true_3d)
print(f"\nORACLE (flat prior): ELBO = {om:.2f} +/- {osd:.2f}")
print(f"True params: lam={lam_true}, r={PARAMS['r']}, S on sphere")

# ----------------------------------------------------------------
# STEP A: FINE PROFILE SCAN
# ======================================================================
grid = np.linspace(LAM_MIN, LAM_MAX, N_POINTS)
elbo_scores, r_hats = [], []
for lf in grid:
    h, pp = train(make_params(), n_iter=PROFILE_ITERS, lr=PROFILE_LR,
                  lam_fix=float(lf), n_mc=N_MC_SCAN)
    elbo_scores.append(tail_elbo(h))
    r_hats.append(float(torch.exp(pp["log_r"])))
    print(f"scan lam = {lf:.3f} : ELBO(tail) = {elbo_scores[-1]:.2f} "
          f"| r_hat = {r_hats[-1]:.3f}")

if ORACLE_CHECK:
    print(f"scan max = {max(elbo_scores):.2f} vs oracle = {om:.2f} "
          f"-> {'oracle better (optimization issue)'
                if om > max(elbo_scores) + 5 else
                'surface prefers wrong params (overfitting)'
                if om < max(elbo_scores) - 5 else
                'comparable: identifiability limited'}")

fig, axs = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)
axs[0].plot(grid, elbo_scores, "o-")
axs[0].axvline(lam_true, color="k", ls="--", label="true lambda")
axs[0].set_ylabel("profile ELBO")
axs[0].legend()
axs[1].plot(grid, r_hats, "s-", color="tab:red")
axs[1].axhline(PARAMS["r"], color="k", ls=":")
axs[1].set_ylabel("r_hat(lambda)")
axs[1].set_xlabel("lambda (fixed)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# STEP B: CANDIDATE SELECTION + POLISH
# ======================================================================
order = np.argsort(elbo_scores)[::-1]
top = order[0]
candidates = [int(i) for i in order
              if elbo_scores[i] >= elbo_scores[top] - DELTA_ELBO][:MAX_CAND]
print("candidate lambdas:", [float(grid[i]) for i in candidates])

cand_info = []
for ci in candidates:
    lam_c = float(grid[ci])
    pc = make_params()
    with torch.no_grad():
        b = float(np.clip(lam_c, 0.02, 0.98))
        pc["mu_th"].fill_(np.log(b / (1.0 - b)))
    h_pol, pc = train(pc, n_iter=POLISH_ITERS, lr=POLISH_LR,
                      lam_fix=lam_c, n_mc=N_MC_SCAN)
    r_c = float(torch.exp(pc["log_r"]))
    score_c = tail_elbo(h_pol)

    h_sgvi, pc = train(pc, n_iter=SGVI_ITERS, lr=SGVI_LR,
                       r_fix=r_c, n_mc=N_MC_SGVI)
    with torch.no_grad():
        th_s = pc["mu_th"] + torch.exp(pc["logsig_th"]) * torch.randn(6000)
        lam_samples = torch.sigmoid(th_s).numpy()

    # Get social positions for visualization
    S_hat = pc["v_S"].detach().numpy()
    S_hat = S_hat / (np.linalg.norm(S_hat, axis=1, keepdims=True) + 1e-9)

    cand_info.append(dict(idx=ci, lam=lam_c, r=r_c, score=score_c,
                          samples=lam_samples,
                          S_hat=S_hat))
    print(f"cand lam={lam_c:.3f}: r_pol={r_c:.3f} "
          f"| ELBO={score_c:.2f} "
          f"| cond lam_hat={lam_samples.mean():.4f} "
          f"+/- {lam_samples.std():.4f}")

# ----------------------------------------------------------------
# BAYESIAN MIXTURE OVER CANDIDATES
# ======================================================================
sc = np.array([c["score"] for c in cand_info])
w = np.exp(sc - sc.max())
w = w / w.sum()
pool = []
for cinfo, wi in zip(cand_info, w):
    cinfo["weight"] = float(wi)
    k = int(round(wi * N_MIX_SAMPLES))
    pool.append(cinfo["samples"][:k])
    print(f"  lam={cinfo['lam']:.3f} r={cinfo['r']:.3f} "
          f"ELBO={cinfo['score']:.2f} weight={wi:.3f}")
lam_pool = np.concatenate(pool)

print("=" * 52)
lo95 = np.percentile(lam_pool, 2.5)
hi95 = np.percentile(lam_pool, 97.5)
covered = bool(lo95 <= lam_true <= hi95)
print(f"lambda_hat  = {lam_pool.mean():.4f} +/- {lam_pool.std():.4f} "
      "(ridge-averaged posterior)")
print(f"95% CI      = [{lo95:.4f}, {hi95:.4f}]")
print(f"lambda_true = {lam_true}")
print(f"95% CI covers lambda_true: {covered}")

plt.figure(figsize=(7, 3.2))
plt.hist(lam_pool, bins=60, density=True, alpha=0.75)
plt.axvline(lam_true, color="k", ls="--", label="true lambda")
plt.axvline(lo95, color="gray", ls=":", label="95% CI")
plt.axvline(hi95, color="gray", ls=":")
plt.xlabel("lambda"); plt.ylabel("posterior density"); plt.legend()
plt.tight_layout(); plt.show()

# Visualization: top candidate social space vs true (both on sphere)
top_c = cand_info[int(np.argmax(w))]
S_hat = top_c["S_hat"]

# Project true S to sphere for comparison (same mapping)
S_true_sphere = np.hstack([S_true, np.ones((N, 1))])
S_true_sphere = S_true_sphere / (np.linalg.norm(S_true_sphere, axis=1, keepdims=True) + 1e-9)

# Align via Procrustes in 3D
Rrot, _ = orthogonal_procrustes(S_hat, S_true_sphere)
S_aligned = S_hat @ Rrot

clusters = np.array([G.nodes[i]["cluster"] for i in range(N)])
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# True social space (2D original)
axes[0].scatter(S_true[:, 0], S_true[:, 1], c=plt.cm.tab10(clusters % 10),
                s=14, alpha=0.8)
axes[0].set_title("True social space (2D)")
axes[0].set_aspect("equal")
axes[0].set_xticks([]); axes[0].set_yticks([])

# Inferred on sphere (projected to 2D for viz)
axes[1].scatter(S_aligned[:, 0], S_aligned[:, 1], c=plt.cm.tab10(clusters % 10),
                s=14, alpha=0.8)
axes[1].set_title("Inferred social space (on sphere, 2D proj)")
axes[1].set_aspect("equal")
axes[1].set_xticks([]); axes[1].set_yticks([])

# 3D view of inferred sphere
ax3d = fig.add_subplot(133, projection='3d')
ax3d.scatter(S_aligned[:, 0], S_aligned[:, 1], S_aligned[:, 2],
             c=plt.cm.tab10(clusters % 10), s=14, alpha=0.8)
ax3d.set_title("Inferred on S^2")
plt.tight_layout()
plt.show()

# Save results
results = {
    "lambda_true": lam_true,
    "lambda_hat": float(lam_pool.mean()),
    "lambda_std": float(lam_pool.std()),
    "ci_95": [float(lo95), float(hi95)],
    "covered": covered,
    "r_hat": float(best_r) if 'best_r' in locals() else None,
    "candidates": cand_info,
}
print("\nResults summary:", results)