1. **Data Preprocessing and GARCH-Filtering**:
   - Load the `returns.csv` dataset and ensure the index is set to datetime objects.
   - Perform an Augmented Dickey-Fuller (ADF) test on each asset return series to confirm stationarity.
   - Implement a rolling-window GARCH(1,1) estimation (using only data available up to time $t$ to avoid look-ahead bias) to generate one-step-ahead conditional volatility forecasts ($\sigma_{i,t}$).
   - Standardize returns by their respective GARCH-forecasted conditional standard deviations to create "GARCH-filtered" innovations for subsequent factor extraction.

2. **Rolling-Window Factor Extraction**:
   - Implement a 60-day rolling window approach across the 1,000-day panel.
   - Within each window, perform Principal Component Analysis (PCA) on the GARCH-filtered returns to extract the first two principal components (latent market and sector factors).
   - Calculate factor loadings ($\beta_{market}, \beta_{sector}$) for each asset by regressing the filtered returns onto the extracted components.

3. **Factor-Based Covariance Matrix Construction**:
   - Construct the factor-based covariance matrix $\Sigma_{factor, t} = B_t \Omega_t B_t^T + \Psi_t$.
   - Define $\Omega_t$ as the covariance matrix of the extracted factors.
   - Define $\Psi_t$ as a diagonal matrix of idiosyncratic variances, calculated as the variance of the residuals from the factor regression, ensuring these are constrained to be positive and scaled by the GARCH-forecasted conditional variances to reflect time-varying idiosyncratic noise.

4. **Shrinkage-Based Covariance Matrix Construction**:
   - Compute the sample covariance matrix $S_t$ for each 60-day window.
   - Apply the Ledoit-Wolf shrinkage estimator using a "Constant Correlation" target, where the target matrix $F$ has elements $F_{ij} = \rho_{ij} \sqrt{\sigma_i^2 \sigma_j^2}$, with $\rho_{ij}$ being the average pairwise correlation and $\sigma_i^2$ being the GARCH-forecasted conditional variance.

5. **Minimum Variance Portfolio (MVP) Construction**:
   - For each time step $t$, calculate the optimal weights for the MVP using both the factor-based and shrinkage-based covariance matrices.
   - Compute weights for two scenarios: (a) Long-only constraint (weights $\ge 0$, sum = 1) and (b) Unconstrained (allowing short selling, sum = 1) to isolate the impact of the covariance estimator from portfolio constraints.

6. **Performance Evaluation**:
   - Calculate the realized portfolio variance for the subsequent day $t+1$ for all strategies.
   - Compute the Information Ratio of the factor-based strategy relative to the shrinkage-based strategy.
   - Assess portfolio stability by calculating the sum of absolute changes in weights ($\sum |w_t - w_{t-1}|$) for both methods.

7. **Statistical Significance Testing**:
   - Perform a Diebold-Mariano test to determine if the difference in realized variance between the factor-based and shrinkage-based portfolios is statistically significant.
   - Compare the results across both the long-only and unconstrained portfolio regimes.

8. **Sensitivity Analysis on Idiosyncratic Volatility**:
   - Calculate the cross-sectional average and the maximum idiosyncratic volatility across the panel for each time step.
   - Perform a rolling-window regression of the performance gap (difference in realized variance) against these idiosyncratic volatility metrics.
   - Identify the threshold of idiosyncratic volatility where the factor-based covariance reconstruction begins to underperform the shrinkage-based approach.