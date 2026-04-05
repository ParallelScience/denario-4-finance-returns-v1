# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
import os

def run_analysis():
    file_path = "/home/node/work/projects/finance_returns_v1/returns.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    tickers = df.columns.tolist()
    tech_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    non_tech_tickers = [t for t in tickers if t not in tech_tickers]
    n_obs = len(df)
    window = 60
    factor_covs = []
    shrinkage_covs = []
    cond_factor = []
    cond_shrink = []
    for i in range(window, n_obs):
        window_data = df.iloc[i-window:i]
        garch_vols = []
        filtered_returns = pd.DataFrame(index=window_data.index, columns=tickers)
        for ticker in tickers:
            model = arch_model(window_data[ticker], vol="Garch", p=1, q=1, rescale=False)
            res = model.fit(disp="off")
            garch_vols.append(res.conditional_volatility.iloc[-1])
            filtered_returns[ticker] = window_data[ticker] / res.conditional_volatility
        garch_vols = np.array(garch_vols)
        pca = PCA(n_components=1)
        market_factor = pca.fit_transform(filtered_returns)
        sector_factor = filtered_returns[tech_tickers].mean(axis=1) - filtered_returns[non_tech_tickers].mean(axis=1)
        sector_factor = sector_factor.values.reshape(-1, 1)
        sector_factor -= (np.dot(sector_factor.T, market_factor) / np.dot(market_factor.T, market_factor)) * market_factor
        factors = np.hstack([market_factor, sector_factor])
        loadings = []
        idiosyncratic_vars = []
        for ticker in tickers:
            y = filtered_returns[ticker].values
            beta = np.linalg.lstsq(factors, y, rcond=None)[0]
            loadings.append(beta)
            resid = y - np.dot(factors, beta)
            idiosyncratic_vars.append(np.var(resid))
        loadings = np.array(loadings)
        factor_cov = np.dot(loadings, loadings.T) + np.diag(idiosyncratic_vars)
        scaled_factor_cov = np.outer(garch_vols, garch_vols) * factor_cov
        lw = LedoitWolf().fit(filtered_returns)
        scaled_shrink_cov = np.outer(garch_vols, garch_vols) * lw.covariance_
        factor_covs.append(scaled_factor_cov)
        shrinkage_covs.append(scaled_shrink_cov)
        cond_factor.append(np.linalg.cond(scaled_factor_cov))
        cond_shrink.append(np.linalg.cond(scaled_shrink_cov))
    np.savez(os.path.join("data", "covariance_analysis.npz"), factor_covs=np.array(factor_covs), shrinkage_covs=np.array(shrinkage_covs), cond_factor=np.array(cond_factor), cond_shrink=np.array(cond_shrink))
    print("Analysis complete. Results saved to data/covariance_analysis.npz")
    print("Mean condition number (Factor): " + str(np.mean(cond_factor)))
    print("Mean condition number (Shrinkage): " + str(np.mean(cond_shrink)))

if __name__ == "__main__":
    run_analysis()