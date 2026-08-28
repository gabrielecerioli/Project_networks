import os, time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt

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
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.dpi": 150
})

epsilons = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
m_lam = np.array([0.051186, 0.053127, 0.068040, 0.061824, 0.073915, 0.104156, 0.144787, 0.163456])
s_lam = np.array([0.060089, 0.062387, 0.070229, 0.066647, 0.074963, 0.089579, 0.105300, 0.125380]) / np.sqrt(15)

m_rho = np.array([0.787629, 0.758869, 0.738079, 0.701665, 0.707002, 0.668851, 0.603833, 0.530058])
s_rho = np.array([0.266452, 0.275708, 0.292262, 0.308754, 0.307250, 0.307017, 0.357964, 0.343832]) / np.sqrt(15)

m_r = np.array([0.897677, 0.999522, 1.561504, 2.094422, 2.649615, 5.260302, 7.029196, 10.211057])
s_r = np.array([2.865399, 2.844009, 3.177372, 3.616310, 3.754519, 6.046574, 7.898311, 10.922105]) / np.sqrt(15)

fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))

# (a) Parameter Estimation Error
ax1 = axes[0, 0]
ax1.plot(epsilons * 100, m_lam, marker="o", color="#1d4ed8", lw=2.2, markersize=5, label=r"Mean Error $|\hat{\lambda} - \lambda_{\mathrm{true}}|$")
ax1.fill_between(epsilons * 100, np.maximum(0, m_lam - 1.96*s_lam), m_lam + 1.96*s_lam, color="#3b82f6", alpha=0.20, label=r"$95\%$ Confidence Interval")
ax1.axvline(20, color="#d97706", ls=":", lw=1.5, label=r"Robust Tolerance Horizon ($\epsilon = 20\%$)")
ax1.axvline(30, color="#dc2626", ls="--", lw=1.5, alpha=0.85, label=r"Breakdown Threshold ($\epsilon = 30\%$)")
ax1.set_xlabel(r"Injected Background Random Noise $\epsilon \, [\%]$")
ax1.set_ylabel(r"Parameter Estimation Error $|\hat{\lambda} - \lambda_{\mathrm{true}}|$")
ax1.set_title("(a) Parameter Estimation Error vs. Noise Level", loc="left", fontweight="bold")
ax1.grid(True); ax1.legend(loc="upper left")

# (b) Latent Social Metric Fidelity
ax2 = axes[0, 1]
ax2.plot(epsilons * 100, m_rho, marker="s", color="#047857", lw=2.2, markersize=5, label=r"Reconstruction Fidelity $\rho_{\mathrm{soc}}$")
ax2.fill_between(epsilons * 100, np.maximum(0, m_rho - 1.96*s_rho), np.minimum(1.0, m_rho + 1.96*s_rho), color="#10b981", alpha=0.20, label=r"$95\%$ Confidence Interval")
ax2.axvline(20, color="#d97706", ls=":", lw=1.5, label=r"Robust Tolerance Horizon ($\epsilon = 20\%$)")
ax2.axvline(30, color="#dc2626", ls="--", lw=1.5, alpha=0.85, label=r"Breakdown Threshold ($\epsilon = 30\%$)")
ax2.set_xlabel(r"Injected Background Random Noise $\epsilon \, [\%]$")
ax2.set_ylabel(r"Social Metric Correlation $\rho_{\mathrm{soc}}$")
ax2.set_title("(b) Latent Social Space Recovery Fidelity", loc="left", fontweight="bold")
ax2.grid(True); ax2.legend(loc="lower left")

# (c) Physical Interaction Scale Error
ax3 = axes[1, 0]
ax3.plot(epsilons * 100, m_r, marker="^", color="#7c3aed", lw=2.2, markersize=5, label=r"Relative Spatial Scale Error $|\hat{r} - r_{\mathrm{true}}| / r_{\mathrm{true}}$")
ax3.fill_between(epsilons * 100, np.maximum(0, m_r - 1.96*s_r), m_r + 1.96*s_r, color="#8b5cf6", alpha=0.20)
ax3.axvline(20, color="#d97706", ls=":", lw=1.5, label=r"Robust Tolerance Horizon ($\epsilon = 20\%$)")
ax3.axvline(30, color="#dc2626", ls="--", lw=1.5, alpha=0.85, label=r"Breakdown Threshold ($\epsilon = 30\%$)")
ax3.set_xlabel(r"Injected Background Random Noise $\epsilon \, [\%]$")
ax3.set_ylabel(r"Relative Spatial Error $|\hat{r} - r_{\mathrm{true}}| / r_{\mathrm{true}}$")
ax3.set_title("(c) Physical Interaction Reach Scale Recovery", loc="left", fontweight="bold")
ax3.grid(True); ax3.legend(loc="upper left")

# (d) Signal Retention / Normalized Resilience Curve
ax4 = axes[1, 1]
norm_fidelity = m_rho / m_rho[0]
norm_param_stability = 1.0 - (m_lam - m_lam[0]) / (1.0 - m_lam[0])
ax4.plot(epsilons * 100, norm_fidelity * 100, marker="d", color="#0284c7", lw=2.2, label=r"Latent Structure Retention $[\%]$")
ax4.plot(epsilons * 100, np.maximum(0, norm_param_stability * 100), marker="v", color="#ea580c", lw=2.2, ls="--", label=r"Mixing Parameter Stability $[\%]$")
ax4.axhline(80, color="#475569", ls=":", lw=1.2, label=r"$80\%$ Performance Retention Baseline")
ax4.axvline(20, color="#d97706", ls=":", lw=1.5, label=r"Robust Tolerance Horizon ($\epsilon = 20\%$)")
ax4.set_xlabel(r"Injected Background Random Noise $\epsilon \, [\%]$")
ax4.set_ylabel(r"Normalized Pipeline Resilience $[\%]$")
ax4.set_title("(d) Normalized Pipeline Resilience under Noise", loc="left", fontweight="bold")
ax4.grid(True); ax4.legend(loc="lower left")

plt.tight_layout()
plt.savefig(f"{output_dir}/test_epsilon_noise_plot.png", dpi=150)
print("Noise dashboard plot saved.")
