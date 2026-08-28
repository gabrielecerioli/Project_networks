import os, time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

os.makedirs("../Robustness_tests", exist_ok=True)

# LaTeX-style font formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13.5,
    "axes.edgecolor": "#1e293b",
    "axes.linewidth": 1.0,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})

print("Testing save logic to ../Robustness_tests...")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([0, 1], [0, 1])
ax.set_title("(a) Test Plot", loc="left", fontweight="bold")
plt.savefig("../Robustness_tests/test_plot.pdf")
plt.savefig("../Robustness_tests/test_plot.png", dpi=150)
print("Files saved successfully.")
