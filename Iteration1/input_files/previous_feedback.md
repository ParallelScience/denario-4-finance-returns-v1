The current analysis plan is technically sound in its GARCH-filtering approach but suffers from significant methodological risks and potential over-engineering.

**1. Methodological Weakness: PCA on Filtered Returns**
You are performing PCA on GARCH-standardized returns. While this removes heteroskedasticity, it fundamentally alters the covariance structure you are trying to recover. PCA on standardized returns (correlation matrix) is not equivalent to PCA on the covariance matrix. If the goal is to reconstruct the covariance matrix $\Sigma_t$, you must ensure the factor loadings and idiosyncratic variances are re-scaled by the GARCH forecasts *after* the PCA extraction. Your current plan to scale $\Psi_t$ is correct, but ensure the factor-based covariance $\Sigma_{factor, t} = B_t \Omega_t B_t^T + \Psi_t$ is explicitly reconstructed using the *unscaled* factor loadings derived from the standardized space.

**2. Over-reliance on Rolling-Window PCA**
PCA is notoriously unstable in small windows (60 days for 10 assets). With only 60 observations, the "latent factors" will capture noise as much as signal, especially for TSLA. 
*   **Actionable Improvement:** Instead of pure PCA, consider a **Constrained Factor Model** where you pre-specify the market factor (e.g., equal-weighted index) and a sector factor (e.g., Tech vs. Non-Tech). This reduces the degrees of freedom and prevents the "rotation" of factors from changing arbitrarily across windows, which currently likely drives your portfolio turnover.

**3. The "Threshold" Fallacy**
Your plan to identify a threshold of idiosyncratic volatility where factor-based models fail is interesting but potentially confounded. If the factor-based model fails, it is likely due to the *inability of the 2-factor model to span the asset space* (model misspecification) rather than just idiosyncratic volatility. 
*   **Actionable Improvement:** Calculate the "Explained Variance Ratio" of your 2-factor model at each step. If the performance gap correlates more strongly with low explained variance than with high idiosyncratic volatility, your conclusion should shift from "volatility threshold" to "model span limitations."

**4. Redundant Complexity in MVP**
You are testing both Long-Only and Unconstrained MVP. Given the synthetic nature of this data (which includes a negative-return asset, XOM), the Unconstrained MVP will likely take massive short positions in XOM to hedge, which is a mathematical artifact rather than a financial insight.
*   **Actionable Improvement:** Focus primarily on the Long-Only constraint. It is more robust and reflects realistic institutional constraints. If you keep the unconstrained version, use it only as a diagnostic for the stability of the covariance matrix (e.g., check the condition number of $\Sigma_t$), not as a primary performance metric.

**5. Missing Diagnostic**
You are missing a comparison of the *condition number* of the shrinkage-based vs. factor-based matrices. Shrinkage is specifically designed to improve the condition number (invertibility). If your factor-based matrix is ill-conditioned, your MVP weights will be hyper-sensitive to estimation error, explaining any performance degradation. Compare the condition numbers before concluding that the factor model is "failing."

**Summary for next iteration:**
Simplify the factor extraction by moving away from pure PCA toward a pre-specified factor structure to reduce noise. Prioritize the condition number of the resulting covariance matrix as a primary diagnostic for why the factor-based approach might underperform shrinkage.