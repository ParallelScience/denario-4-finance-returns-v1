1. **Dynamic Factor Extraction via PCA**:
   - Perform a rolling-window PCA on the 60-day GARCH-filtered returns.
   - Extract the first two principal components ($PC_1, PC_2$) at each time step $t$ to serve as latent market and sector factors.
   - Ensure sign consistency of factors across windows by aligning current eigenvectors with the previous window's loadings to prevent artificial volatility.

2. **Regularized Factor Loading Estimation**:
   - Estimate the factor loading matrix $B_t$ by regressing GARCH-filtered returns onto $PC_1$ and $PC_2$ using weighted Ridge regression.
   - Scale the Ridge penalty $\lambda$ by the inverse of the asset-specific GARCH-forecasted idiosyncratic variance to prioritize stability for high-beta/high-volatility assets like TSLA.
   - Tune $\lambda$ using a rolling time-series cross-validation approach within the 60-day window to minimize out-of-sample prediction error without look-ahead bias.

3. **Hybrid Covariance Construction**:
   - Calculate the residual matrix $E_t = R_t - B_t F_t$.
   - Estimate the idiosyncratic covariance matrix $\Psi_{shrink, t}$ by applying the Ledoit-Wolf shrinkage estimator to $E_t$, using a constant correlation target to capture residual cross-sectional dependencies.
   - Construct the hybrid covariance matrix as $\Sigma_{hybrid, t} = \text{diag}(\sigma_t) (B_t \Omega_t B_t^T + \Psi_{shrink, t}) \text{diag}(\sigma_t)$, where $\Omega_t$ is the factor covariance and $\sigma_t$ are the GARCH-forecasted volatilities.

4. **Benchmark Covariance Estimation**:
   - Compute the standard Ledoit-Wolf shrinkage covariance matrix $\Sigma_{LW, t}$ using the same GARCH-filtered returns to isolate the benefit of structural factor decomposition.
   - Include an Equal-Weight (1/N) portfolio as a "no-information" baseline to evaluate the value-add of the MVP optimization process.

5. **Minimum Variance Portfolio (MVP) Optimization**:
   - Solve for weights $w_t$ minimizing $w^T \Sigma w$ subject to $\sum w = 1$.
   - Execute two test cases: unconstrained optimization and a "No-Shorting" constraint ($w \ge 0$) to ensure results are not driven by extreme short positions in assets like XOM or JNJ.
   - Re-estimate GARCH parameters ($\alpha, \beta, \omega$) at each step $t$ using only historical data to avoid look-ahead bias.

6. **Performance and Stability Metrics**:
   - Calculate realized portfolio variance at $t+1$, portfolio turnover, and the Herfindahl-Hirschman Index (HHI) of weights to measure portfolio concentration.
   - Track the condition number of the covariance matrices and the "Factor Stability" (correlation of $B_t$ between consecutive windows) to identify structural breaks.

7. **Sensitivity Analysis to Idiosyncratic Volatility**:
   - Segment the 1,000-day period into regimes based on the GARCH-forecasted idiosyncratic variance of TSLA.
   - Compare the performance gap (Realized Variance$_{LW}$ - Realized Variance$_{Hybrid}$) across these regimes to identify the threshold where factor-based reconstruction fails due to estimation noise.

8. **Statistical Significance Testing**:
   - Perform a Diebold-Mariano test on the daily realized squared returns of the hybrid vs. benchmark portfolios.
   - Aggregate results to determine if the hybrid factor-shrinkage approach provides a statistically robust improvement in risk management under GARCH-induced heteroskedasticity.