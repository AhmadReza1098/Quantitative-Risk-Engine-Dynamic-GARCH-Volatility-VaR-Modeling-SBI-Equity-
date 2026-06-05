# Quantitative Risk Engine: Dynamic GARCH Volatility & VaR Modeling (SBI Equity)

![Quantitative Risk Dashboard](q_risk_dashboard.png) *(Note: Ensure your dashboard image is saved as `q_risk_dashboard.png` in your repository folder)*

This repository demonstrates the end-to-end engineering of a Basel-compliant market risk engine utilizing the GARCH family of models. Applied to 28 years of high-frequency daily equity data for State Bank of India (NSE: SBIN.NS), this project covers every step from data acquisition and empirical volatility diagnostics to model specification, regulatory backtesting, and dynamic Value at Risk (VaR) forecasting. 

---

## Table of Contents

- [Overview](#overview)
- [Data Source](#data-source)
- [Requirements](#requirements)
- [Workflow](#workflow)
- [Key Business & Quantitative Results](#key-business--quantitative-results)
- [How to Replicate](#how-to-replicate)
- [References](#references)

---

## Overview

Financial markets rarely exhibit constant volatility. Periods of calm are often followed by turbulent crashes, and market shocks leave a lasting "memory" on risk levels. Relying on standard, smooth Normal distributions to predict risk in these conditions is a recipe for catastrophic failure. 

This project moves beyond theoretical forecasting to build a usable risk management tool. By empirically diagnosing volatility clustering and extreme tail risks (leptokurtosis), this script successfully generates dynamic 99% Parametric VaR and Expected Shortfall (CVaR) limits that pass strict regulatory validation.

---

## Data Source

- **Asset:** State Bank of India (Ticker: SBIN.NS)
- **Frequency:** Daily Close Prices
- **Timeframe:** 1996-01-01 to 2026-05-25 (28 Years / ~7,600 Trading Days)
- **Source:** Yahoo Finance (via local CSV to bypass network blocks)

---

## Requirements

The pipeline is built entirely in R. Install the following packages to execute the models and generate the risk dashboard:

```R
install.packages(c("quantmod", "rugarch", "PerformanceAnalytics", "forecast", "urca", "ggplot2", "tseries", "MTS", "FinTS", "reshape2", "patchwork", "zoo", "xts"))
```

## Workflow

### 1. Data Engineering & Stationarity
* Acquired and cleaned 28 years of daily price data, handling missing values and market holidays.
* Converted non-stationary price action into stationary log returns.
* Executed Augmented Dickey-Fuller (ADF) tests to confirm structural stationarity prior to modeling.

### 2. Empirical Volatility Diagnostics
* Mapped squared returns to visually confirm severe volatility clustering (market memory).
* Applied strict ARCH-LM testing to mathematically prove the presence of conditional heteroskedasticity.

### 3. The "Horse Race" (Model Selection)
* Specified, fitted, and evaluated multiple volatility structures: **sGARCH(1,1), sGARCH(1,2), GARCH-M, TGARCH,** and **EGARCH**.
* Utilized Information Criteria (AIC/BIC) and post-estimation diagnostics (Nyblom stability, Ljung-Box tests) to formally reject asymmetric models in favor of a higher-order **sGARCH(1,2)** structure.

### 4. Risk Management (VaR & CVaR)
* Extracted dynamic conditional volatility (sigma) to calculate rolling 1-Day 99% Parametric Value at Risk (VaR) limits.
* Computed Conditional VaR (Expected Shortfall) to quantify average expected losses beyond the VaR threshold.

### 5. Regulatory Backtesting & The "Fat Tail" Pivot
* Executed a Basel-standard Kupiec Proportion of Failures (POF) test.
* **The Diagnosis:** The initial Normal distribution model severely underestimated market crashes (failing the Kupiec test with 105 actual breaches vs. 76 expected).
* **The Fix:** Re-specified the optimal sGARCH(1,2) model using a **Student-t distribution** to capture extreme tail risk, successfully passing regulatory validation.

---

## Key Business & Quantitative Results

* **Superior Volatility Mapping:** The higher-order sGARCH(1,2) successfully captured complex, non-monotonic "zig-zag" variance aftershocks, achieving a highly precise out-of-sample Mean Absolute Error (MAE) of 1.19%.
* **Mathematical Proof of Tail Risk:** The Student-t distribution optimization estimated a shape parameter (degrees of freedom) of **5.47**. In quantitative finance, any shape value below 10 is strict empirical proof of extreme fat tails (leptokurtosis) in the asset.
* **Basel-Compliant Risk Limits:** By upgrading to the Student-t distribution, the VaR limits successfully absorbed extreme market crashes. Actual VaR breaches dropped to **62** (well within the expected limit of 76), achieving a passing Kupiec p-value of **0.0898**, rendering the model regulatory-compliant.
* **Executive Dashboard:** Successfully translated complex econometric outputs into a 4-panel `ggplot2` risk dashboard, displaying the VaR limits, tail distributions, and Basel traffic-light metrics for immediate trading desk utility.

---

## How to Replicate

1. Clone this repository to your local machine.
2. Ensure the `SBIN.NS.csv` data file is in your working directory.
3. Open the main `Emperical Volatility - GARCH.R` script in RStudio.
4. Run the code sequentially. The script will automatically generate the models, run the Kupiec backtests, and output the final 4-panel Quantitative Risk Dashboard in your plots pane.

---

## References

* Engle, R.F. (1982). "Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*.
* Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*.
* Basel Committee on Banking Supervision (BCBS) - Guidelines on Market Risk & Backtesting.
* `rugarch` package CRAN documentation.
