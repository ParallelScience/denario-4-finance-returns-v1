

Iteration 0:
**Summary: Dynamic Factor-Covariance Recovery under GARCH-Induced Heteroskedasticity**

**1. Status & Methodology**
*   **Completed:** Data preprocessing, ADF stationarity verification, rolling-window (60-day) GARCH(1,1) conditional volatility estimation, and PCA-based latent factor extraction (market/sector).
*   **Intermediate Output:** `data/intermediate_data.npz` contains standardized returns, factor loadings, and extracted factors.
*   **Methodological Choice:** Returns were standardized by GARCH-forecasted conditional volatility prior to PCA to isolate systematic signals from time-varying idiosyncratic noise.

**2. Key Findings & Limitations**
*   **Stationarity:** ADF tests confirm all 10 asset return series are stationary.
*   **Computational Constraint:** The current implementation uses a nested loop for GARCH estimation, which is computationally expensive for large panels.
*   **Data Integrity:** GARCH-filtered returns were successfully generated; however, the impact of TSLA’s high idiosyncratic volatility on factor loading stability remains to be quantified in the next phase.

**3. Decisions for Future Experiments**
*   **Covariance Construction:** Proceed with constructing $\Sigma_{factor, t} = B_t \Omega_t B_t^T + \Psi_t$ and the Ledoit-Wolf shrinkage target using the saved intermediate data.
*   **Portfolio Optimization:** Implement MVP construction (long-only vs. unconstrained) to compare realized variance and tracking error.
*   **Sensitivity Analysis:** The performance gap between factor-based and shrinkage-based estimators must be regressed against idiosyncratic volatility metrics to identify the failure threshold for factor-based reconstruction.
*   **Statistical Validation:** Execute Diebold-Mariano tests on realized variance differences to confirm significance.

**4. Open Issues**
*   The threshold of idiosyncratic volatility where factor-based reconstruction fails is currently unknown.
*   Portfolio stability (weight turnover) has not yet been evaluated.
        