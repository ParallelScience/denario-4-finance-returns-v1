# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.decomposition import PCA
import os

def run_preprocessing():
    data_dir = "data/"
    returns = pd.read_csv("returns.csv", index_col=0)
    corr_matrix = returns.corr()
    print("Unconditional Correlation Matrix (first 5x5):")
    print(corr_matrix.iloc[:5, :5])
    filtered_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
    garch_params = {}
    for ticker in returns.columns:
        model = arch_model(returns[ticker], vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off')
        garch_params[ticker] = res.params
        filtered_returns[ticker] = returns[ticker] / res.conditional_volatility
    filtered_returns.to_csv(os.path.join(data_dir, "filtered_returns.csv"))
    window = 60
    factors = []
    prev_components = None
    for i in range(window, len(filtered_returns)):
        window_data = filtered_returns.iloc[i-window:i]
        pca = PCA(n_components=2)
        pca.fit(window_data)
        components = pca.components_
        if prev_components is not None:
            for j in range(2):
                if np.dot(components[j], prev_components[j]) < 0:
                    components[j] *= -1
        factors.append(pca.transform(window_data.iloc[[-1]])[0])
        prev_components = components
    factors_df = pd.DataFrame(factors, index=returns.index[window:], columns=['Factor1', 'Factor2'])
    factors_df.to_csv(os.path.join(data_dir, "latent_factors.csv"))
    print("\nExtracted Factors (first 5 rows):")
    print(factors_df.head())
    print("\nMean Correlation of Returns: " + str(corr_matrix.values[np.triu_indices(10, k=1)].mean()))

if __name__ == '__main__':
    run_preprocessing()