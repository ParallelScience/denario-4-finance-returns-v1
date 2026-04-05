# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
import os

def run_preprocessing():
    data_dir = "data/"
    returns_path = os.path.join(data_dir, "returns.csv")
    df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    print("ADF Test Results (p-values):")
    for col in df.columns:
        adf_res = adfuller(df[col].dropna())
        print(col + ": " + str(round(adf_res[1], 4)))
    n_obs, n_assets = df.shape
    window = 60
    garch_vol = np.full((n_obs, n_assets), np.nan)
    for i in range(n_assets):
        series = df.iloc[:, i]
        for t in range(window, n_obs):
            model = arch_model(series.iloc[:t], vol='Garch', p=1, q=1, rescale=False)
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=1)
            garch_vol[t, i] = np.sqrt(forecast.variance.iloc[-1, 0])
    filtered_returns = df.values / garch_vol
    filtered_returns = np.nan_to_num(filtered_returns)
    loadings = []
    factors = []
    for t in range(window, n_obs):
        window_data = filtered_returns[t-window:t, :]
        pca = PCA(n_components=2)
        pca.fit(window_data)
        components = pca.transform(window_data)
        factors.append(components[-1, :])
        window_loadings = np.zeros((n_assets, 2))
        for i in range(n_assets):
            y = window_data[:, i]
            X = components
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            window_loadings[i, :] = beta
        loadings.append(window_loadings)
    np.savez(os.path.join(data_dir, "intermediate_data.npz"), filtered_returns=filtered_returns, loadings=np.array(loadings), factors=np.array(factors))
    print("Preprocessing complete. Data saved to data/intermediate_data.npz")

if __name__ == '__main__':
    run_preprocessing()