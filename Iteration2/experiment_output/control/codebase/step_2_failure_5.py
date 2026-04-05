# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

def solve_mvp(cov, no_short=False):
    n = cov.shape[0]
    def objective(w):
        return w.T @ cov @ w
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) if no_short else (None, None) for _ in range(n)]
    res = minimize(objective, np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

def run_step_2():
    data_dir = 'data/'
    returns = pd.read_csv('returns.csv', index_col=0)
    filtered_returns = pd.read_csv(os.path.join(data_dir, 'filtered_returns.csv'), index_col=0)
    factors = pd.read_csv(os.path.join(data_dir, 'latent_factors.csv'), index_col=0)
    window = 60
    n_assets = returns.shape[1]
    hybrid_vars = []
    lw_vars = []
    prev_loadings = None
    stability = []
    cond_nums = []
    for i in range(window, len(returns)):
        ret_win = filtered_returns.iloc[i-window:i].copy()
        ret_win.columns = ret_win.columns.astype(str)
        fac_win = factors.iloc[i-window:i]
        loadings = np.zeros((n_assets, 2))
        for j in range(n_assets):
            ridge = Ridge(alpha=0.1)
            ridge.fit(fac_win, ret_win.iloc[:, j])
            loadings[j] = ridge.coef_
        if prev_loadings is not None:
            stability.append(np.corrcoef(loadings.flatten(), prev_loadings.flatten())[0, 1])
        prev_loadings = loadings
        resid = (ret_win - fac_win @ loadings.T).fillna(0)
        resid.columns = resid.columns.astype(str)
        lw = LedoitWolf().fit(resid)
        psi = lw.covariance_
        omega = np.cov(fac_win, rowvar=False)
        sigma_hybrid = loadings @ omega @ loadings.T + psi
        sigma_lw = LedoitWolf().fit(ret_win.fillna(0)).covariance_
        cond_nums.append((np.linalg.cond(sigma_hybrid), np.linalg.cond(sigma_lw)))
        w_h = solve_mvp(sigma_hybrid)
        w_l = solve_mvp(sigma_lw)
        hybrid_vars.append(w_h.T @ sigma_hybrid @ w_h)
        lw_vars.append(w_l.T @ sigma_lw @ w_l)
    print('Mean Realized Variance (Hybrid): ' + str(np.mean(hybrid_vars)))
    print('Mean Realized Variance (LW): ' + str(np.mean(lw_vars)))
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(cond_nums)
    ax[0].set_title('Condition Number of Covariance Matrices')
    ax[0].legend(['Hybrid', 'LW'])
    ax[1].plot(stability)
    ax[1].set_title('Factor Loading Stability (Correlation)')
    plt.tight_layout()
    plot_path = os.path.join(data_dir, 'stability_metrics_2_20231101.png')
    plt.savefig(plot_path, dpi=300)
    print('Plot saved to ' + plot_path)

if __name__ == '__main__':
    run_step_2()