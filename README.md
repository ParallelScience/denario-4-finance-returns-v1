# Finance Returns: Factor Model & Portfolio Analysis

**Scientist:** denario-4 (Denario AI Research Scientist)
**Date:** 2026-04-05
**Status:** Setup complete — pipeline starting

## Dataset

Synthetic daily returns panel for 10 large-cap US equities (2020–2023), generated from a two-factor model with GARCH volatility clustering.

## Dataset: Synthetic Equity Returns Panel

### Overview
A synthetic daily returns panel for 10 large-cap US equities, generated from a two-factor linear model with GARCH volatility clustering, covering 1,000 trading days (2020-01-02 to 2023-11-01).

### Assets
AAPL, MSFT, GOOGL, AMZN, META (technology sector), TSLA (high-beta growth), JPM, GS (financials), XOM (energy), JNJ (healthcare/defensive).

### Data Files
- `returns.csv` — daily log-returns, shape (1000, 10), columns = ticker names, index = date (business days)
- `prices.csv` — price series (base 100), shape (1000, 10)
- `returns.npy` — NumPy structured array with named fields per ticker

### Data Generating Process
1. **Two-factor model**: each asset's return = β_market × market_factor + β_sector × sector_factor + ε_idiosyncratic
   - Market factor: N(0.0003, 0.012) daily — ~7.5% annualised return, 19% vol
   - Sector factor: N(0.0001, 0.008) — separates tech/growth from financials/energy
2. **Volatility clustering (GARCH(1,1))**: TSLA return series has time-varying volatility (α=0.10, β=0.85, ω=1e-5)
3. **Idiosyncratic shocks**: asset-specific Gaussian noise with volatilities ranging from 1.0% (JNJ) to 3.0% (TSLA)

### Key Statistics (annualised)
| Ticker | Return | Volatility | Sharpe |
|--------|--------|-----------|--------|
| AAPL   | 31.7%  | 33.9%     | 0.94   |
| MSFT   | 22.1%  | 30.3%     | 0.73   |
| GOOGL  |  9.5%  | 29.8%     | 0.32   |
| AMZN   | 12.0%  | 36.5%     | 0.33   |
| META   | 19.4%  | 39.6%     | 0.49   |
| TSLA   | 52.8%  | 67.0%     | 0.79   |
| JPM    | 12.6%  | 25.7%     | 0.49   |
| GS     |  6.1%  | 27.7%     | 0.22   |
| XOM    |-11.8%  | 22.8%     |-0.52   |
| JNJ    | -3.2%  | 20.8%     |-0.15   |

### Cross-sectional structure
- Tech stocks are highly correlated (pairwise ρ ≈ 0.45–0.55)
- Tech vs financials/energy have lower or negative correlations via the sector factor
- True market β ranges from 0.6 (JNJ) to 1.8 (TSLA)

### Suggested Research Directions
- Factor model estimation and latent factor recovery
- Portfolio optimisation (mean-variance, minimum variance, maximum Sharpe)
- Volatility modelling and forecasting (GARCH on individual series)
- Risk decomposition: systematic vs idiosyncratic risk
- Covariance matrix estimation (shrinkage, factor-based)
- Cross-sectional return predictability from betas
