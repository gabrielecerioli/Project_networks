"""
Test analysis of covered vs missed networks from the 100-run Monte Carlo simulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def analyze_covered_vs_missed(results_list):
    df = pd.DataFrame(results_list)
    
    # Add derived metrics
    df["z_score"] = (df["lam_hat"] - df["lam_true"]) / df["lam_std"]
    df["ci_width"] = [ci[1] - ci[0] for ci in df["ci_95"]]
    df["status"] = df["covered"].map({True: "Covered (95% CI)", False: "Missed"})
    
    print("=== MISSED NETWORKS DETAILS ===")
    missed_df = df[~df["covered"]]
    print(missed_df[["id", "N", "lam_true", "lam_hat", "ci_95", "r_true", "r_hat", "err_lam", "z_score"]])
    
    # Statistical comparison (Mann-Whitney U tests)
    features = ["N", "lam_true", "r_true", "err_lam", "ci_width"]
    p_values = {}
    covered = df[df["covered"]]
    missed = df[~df["covered"]]
    
    for feat in features:
        if len(missed) > 0 and len(covered) > 0:
            stat, pval = stats.mannwhitneyu(covered[feat], missed[feat])
            p_values[feat] = pval
            print(f"{feat:15s} | Covered Mean: {covered[feat].mean():.3f} | Missed Mean: {missed[feat].mean():.3f} | p-val: {pval:.4f}")

if __name__ == "__main__":
    pass
