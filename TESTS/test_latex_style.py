import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# Configure publication-grade LaTeX-style plotting aesthetics
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "axes.edgecolor": "#334155",
    "axes.linewidth": 1.0,
    "grid.color": "#cbd5e1",
    "grid.linestyle": "--",
    "grid.alpha": 0.5
})

print("Testing LaTeX styling configuration...")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([0, 1], [0, 1], label=r"$\mathcal{L}(\hat{\lambda}) - \mathcal{L}^*$")
ax.set_xlabel(r"$\eta_{\mathrm{Adam}}$")
ax.set_ylabel(r"$|\hat{\lambda} - \lambda_{\mathrm{true}}|$")
ax.set_title(r"\textbf{Hyperparameter Convergence Test}")
ax.legend()
plt.tight_layout()
plt.savefig("test_latex_style.png", dpi=200)
print("Saved test_latex_style.png successfully.")
