1. **Data Preprocessing and GARCH-Filtering**:
   - Load the `returns.csv` dataset and ensure the index is set to datetime objects.
   - Implement a rolling-window GARCH(1,1) estimation (using only data available up to time $t$) to generate one-step-ahead conditional volatility forecasts ($\sigma_{i,t}$) for each asset.
   - Standardize returns by their respective GARCH-forecasted conditional standard deviations to create "GARCH-filtered" innovations.

2. **Orthogonalized Factor Model Specification**:
   - Define the Market Factor as the first principal component of the 10 assets within the 60-day window to capture the primary systematic movement.
   - Define the Sector Factor as a "Long-Short" factor (Long Tech: AAPL, MSFT, GOOGL, AMZN, META; Short Non-Tech: JPM, GS, XOM, JNJ).
   - Orthogonalize the Sector Factor against the Market Factor using Gram-Schmidt to ensure the two factors are uncorrelated, simplifying the factor covariance matrix $\Omega_t$ to a diagonal structure.

3. **Factor-Based Covariance Matrix Construction**:
   - For each 60-day window, estimate factor loadings ($\beta_{market}, \beta_{sector}$) via OLS regression of the GARCH-filtered returns onto the orthogonalized factors.
   - Reconstruct the covariance matrix as $\Sigma_{factor, t} = \text{diag}(\sigma_t) (B_t \Omega_t B_t^T + \Psi_t) \text{diag}(\sigma_t)$, where $\sigma_t$ is the vector of GARCH-forecasted volatilities, $B_t$ is the matrix of loadings, $\Omega_t$ is the diagonal factor covariance, and $\Psi_t$ is the diagonal matrix of idiosyncratic variances from the residuals.

4. **Shrinkage-Based Covariance Matrix Construction**:
   - Compute the sample covariance matrix $S_t$ for each 60-day window using the GARCH-filtered innovations.
   - Apply the Ledoit-Wolf shrinkage estimator using a "Constant Correlation" target, where the target matrix $F$ elements are $F_{ij} = \rho_{avg} \sqrt{\sigma_i^2 \sigma_j^2}$, with $\rho_{avg}$ calculated from the filtered innovations.
   - Use the analytical Ledoit-Wolf formula to determine the optimal shrinkage intensity $\delta$ for each window.

5. **Diagnostic: Condition Number Analysis**:
   - Calculate the condition number for both $\Sigma_{factor, t}$ and $\Sigma_{shrinkage, t}$ at each time step.
   - Use these values to assess the numerical stability and invertibility of the matrices, providing a diagnostic baseline for potential MVP weight instability.

6. **Minimum Variance Portfolio (MVP) Construction**:
   - Calculate optimal weights for the Long-Only MVP (weights $\ge 0$, sum = 1) using both the factor-based and shrinkage-based covariance matrices.
   - Use the Unconstrained MVP (sum = 1) as a diagnostic tool to evaluate the sensitivity of weights to the covariance structure and the impact of negative-return assets.

7. **Performance Evaluation and Turnover Analysis**:
   - Calculate the realized portfolio variance for the subsequent day $t+1$ for both strategies.
   - Compute the Information Ratio of the factor-based strategy relative to the shrinkage-based strategy.
   - Calculate portfolio turnover ($\sum |w_t - w_{t-1}|$) and assess if performance gains are eroded by rebalancing costs.

8. **Model Span and Performance Attribution**:
   - Calculate the "Explained Variance Ratio" (R-squared) of the 2-factor model at each step.
   - Perform a regression of the performance gap (difference in realized variance) against the Explained Variance Ratio and the condition number.
   - Conclude whether performance degradation is driven by model span limitations or numerical instability, noting the rigidity of the constant correlation shrinkage target.