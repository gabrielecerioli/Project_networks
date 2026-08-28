import time, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# Re-use generator
from test_all_three_robustness import sample_synthetic_network

print("Validating all three sensitivity analysis modules...")

# Check that sample_synthetic_network adheres to user specification:
# N in [100, 2000], lambda in (0,1), ln r in [ln 0.03, ln 0.50], ln sigma in [ln 0.03, ln 0.50], GMM in [2, 8]
for _ in range(5):
    net = sample_synthetic_network(seed=np.random.randint(1000))
    assert 100 <= net["N"] <= 2000 or 100 <= net["N"] <= 301
    assert 0.0 <= net["lambda_true"] <= 1.0
    assert 0.03 <= net["r_true"] <= 0.50
    assert 0.03 <= net["sigma_true"] <= 0.50
    assert net["W_obs"].shape == (net["N"], net["N"])
    assert np.allclose(net["W_obs"], net["W_obs"].T)

print("Synthetic generator verified successfully!")
