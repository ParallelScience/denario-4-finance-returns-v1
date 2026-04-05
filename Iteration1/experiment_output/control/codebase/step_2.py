# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import time

def run_portfolio_analysis():
    data_dir = "data/"
    returns_path = "/home/node/work/projects/finance_returns_v1/returns.csv"
    cov_data = np.load(os.path.join(data_dir, "covariance_analysis.npz"))
    factor_covs = cov_data["factor_covs"]
    shrinkage_covs = cov_data["shrinkage_covs"]
    cond_factor = cov_data["cond_factor"]
    cond_shrink = cov_data["cond_shrink"]
    df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    returns = df.values[60:]
    n_steps = len(factor_covs)
    weights_factor = np.zeros((n_steps, 10))
    weights_shrink = np.zeros((n_steps, 10))
    for i in range(n_steps):
        inv_factor = np.linalg.pinv(factor_covs[i])
        weights_factor[i] = inv_factor.sum(axis=1) / inv_factor.sum()
        inv_shrink = np.linalg.pinv(shrinkage_covs[i])
        weights_shrink[i] = inv_shrink.sum(axis=1) / inv_shrink.sum()
    realized_var_factor = []
    realized_var_shrink = []
    turnover_factor = [0]
    turnover_shrink = [0]
    for i in range(n_steps - 1):
        ret_next = returns[i+1]
        realized_var_factor.append((np.dot(weights_factor[i], ret_next))**2)
        realized_var_shrink.append((np.dot(weights_shrink[i], ret_next))**2)
        if i > 0:
            turnover_factor.append(np.sum(np.abs(weights_factor[i] - weights_factor[i-1])))
            turnover_shrink.append(np.sum(np.abs(weights_shrink[i] - weights_shrink[i-1])))
    r_squared = []
    for i in range(n_steps):
        f_cov = factor_covs[i]
        total_var = np.trace(f_cov)
        resid_var = np.diag(f_cov)[-1]
        r_squared.append(1 - (resid_var / total_var))
    print("Mean Realized Variance (Factor): " + str(np.mean(realized_var_factor)))
    print("Mean Realized Variance (Shrinkage): " + str(np.mean(realized_var_shrink)))
    print("Mean Turnover (Factor): " + str(np.mean(turnover_factor)))
    print("Mean Turnover (Shrinkage): " + str(np.mean(turnover_shrink)))
    info_ratio = (np.mean(realized_var_shrink) - np.mean(realized_var_factor)) / np.std(np.array(realized_var_factor) - np.array(realized_var_shrink))
    print("Information Ratio (Factor vs Shrinkage): " + str(info_ratio))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(realized_var_factor, label="Factor")
    axes[0].plot(realized_var_shrink, label="Shrinkage")
    axes[0].set_title("Realized Portfolio Variance")
    axes[0].legend()
    axes[1].plot(cond_factor, label="Factor")
    axes[1].plot(cond_shrink, label="Shrinkage")
    axes[1].set_title("Condition Number")
    axes[1].legend()
    axes[2].plot(turnover_factor, label="Factor")
    axes[2].plot(turnover_shrink, label="Shrinkage")
    axes[2].set_title("Portfolio Turnover")
    axes[2].legend()
    plt.tight_layout()
    ts = str(int(time.time()))
    plot_path1 = os.path.join(data_dir, "summary_plot_" + ts + ".png")
    plt.savefig(plot_path1, dpi=300)
    print("Saved to " + plot_path1)
    plt.figure(figsize=(10, 5))
    plt.plot(r_squared)
    plt.title("Explained Variance Ratio (R-squared) over time")
    plt.xlabel("Time Step")
    plt.ylabel("R-squared")
    plt.grid(True)
    plt.tight_layout()
    plot_path2 = os.path.join(data_dir, "r_squared_plot_" + ts + ".png")
    plt.savefig(plot_path2, dpi=300)
    print("Saved to " + plot_path2)

if __name__ == "__main__":
    run_portfolio_analysis()