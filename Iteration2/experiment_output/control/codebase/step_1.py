# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.decomposition import PCA

def generate_synthetic_data():
    np.random.seed(42)
    n_days = 1000
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'GS', 'XOM', 'JNJ']
    market_factor = np.random.normal(0.0003, 0.012, n_days)
    sector_factor = np.random.normal(0.0001, 0.008, n_days)
    returns = pd.DataFrame(index=pd.date_range('2020-01-02', periods=n_days), columns=tickers)
    for ticker in tickers:
        beta_m = np.random.uniform(0.6, 1.8)
        beta_s = np.random.uniform(-0.5, 0.5)
        idiosyncratic = np.random.normal(0, 0.02, n_days)
        if ticker == 'TSLA':
            vol = np.zeros(n_days)
            vol[0] = 0.03
            for t in range(1, n_days):
                vol[t] = np.sqrt(1e-5 + 0.10 * (idiosyncratic[t-1]**2) + 0.85 * (vol[t-1]**2))
            idiosyncratic *= (vol / 0.02)
        returns[ticker] = beta_m * market_factor + beta_s * sector_factor + idiosyncratic
    returns.to_csv('returns.csv')
    return returns

def run_preprocessing():
    data_dir = 'data/'
    if not os.path.exists('returns.csv'):
        returns = generate_synthetic_data()
    else:
        returns = pd.read_csv('returns.csv', index_col=0)
    corr_matrix = returns.corr()
    print('Unconditional Correlation Matrix (first 5x5):')
    print(corr_matrix.iloc[:5, :5])
    filtered_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
    for ticker in returns.columns:
        model = arch_model(returns[ticker], vol='Garch', p=1, q=1, rescale=False)
        res = model.fit(disp='off')
        filtered_returns[ticker] = returns[ticker] / res.conditional_volatility
    filtered_returns.to_csv(os.path.join(data_dir, 'filtered_returns.csv'))
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
    factors_df.to_csv(os.path.join(data_dir, 'latent_factors.csv'))
    print('\nExtracted Factors (first 5 rows):')
    print(factors_df.head())
    mean_corr = corr_matrix.values[np.triu_indices(10, k=1)].mean()
    print('\nMean Correlation of Returns: ' + str(mean_corr))

if __name__ == '__main__':
    run_preprocessing()