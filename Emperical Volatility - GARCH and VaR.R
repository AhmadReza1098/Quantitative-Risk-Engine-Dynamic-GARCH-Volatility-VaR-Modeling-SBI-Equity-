
############################################
#Install required packages
# Loading packages
###########################################

# Install necessary packages
install.packages(c("quantmod", "rugarch", "PerformanceAnalytics", "forecast"))
install.packages("urca")

# Load libraries
library(quantmod) # To get the data
library(rugarch) # To estimate the GARCH model
library(PerformanceAnalytics) # For evaluation
library(forecast) # For forecasting 
library(urca) # for unit root
library(ggplot2) # For ploting 
library(tseries) # for unit root
library(MTS) # for ARCH test
library(FinTS) #for function `ArchTest()`

#########################################################
# Downloading data and cleaning the data
########################################################


# Download TCS stock data
getSymbols("SBIN.NS", src = "yahoo", from = "1996-01-01", to = "2026-05-25")

# Taking the concerned data 
wdata <- Cl(SBIN.NS) # taking close price

# examining the data
head(wdata)
start(wdata)
end(wdata)
summary(wdata)

#check for missing values

sum(is.na(wdata))

wdata <- na.omit(wdata)


# Plotting series including ACF and PACF
ggtsdisplay(wdata, plot.type = "partial",smooth = TRUE, theme=theme_classic())
# Visual diagnostics (trending series, slow ACF decay, PACF spike at lag 1) visually confirm a non-stationary I(1) random walk.

#Check for stationarity (ADF-test)
adf <- ur.df(wdata, type = "drift", selectlags = "BIC")
adf
summary(adf)

summary(ur.df(wdata, type = "drift", selectlags = "AIC"))
# ADF test (AIC) yields tau = 0.9196 > -2.86 (5% cv); 
#fails to reject unit root, confirming an I(1) non-stationary process. 
# => H0 hypothesis of non-stationarity is failed to be rejected


#Take log of the prices
rtcs = diff(log(wdata))
rtcs <- rtcs[-1,] # dropping first observation for NA
#Plot the graph
plot(rtcs, type = "l", xlab = "")


#Check for stationarity again for transformed series(ADF-test)
summary(ur.df(rtcs, type = "drift", selectlags = "AIC"))
# ADF test on log returns yields tau = -61.19 < -3.43 (1% cv), 
# conclusively rejecting the unit root and confirming a stationary I(0) process.


# Visualize returns
ts.plot(rtcs, main = "Daily Log Returns", col = "blue")
ggtsdisplay(rtcs, plot.type = "partial",smooth = TRUE, theme=theme_classic())
# Visual diagnostics of log returns confirm mean-stationarity and exhibit distinct volatility clustering, 
#formally justifying GARCH model specification.
# Looks pretty stationary
# ACF, few minor spikes persistent autocorrelation at some lag
# PACF, some minor spike at some lag
# it seems to have AR & MA structure in the mean eqn


##################################################################
# GARCH Modelling
##################################################################

# Check for the presence of ARCH effect 
#ARCH Test
rtcsArchTest <- ArchTest(rtcs, lags=1, demean=TRUE)
rtcsArchTest
# ARCH-LM test strongly rejects the null of homoskedasticity (p < 2.2e-16), 
# confirming significant ARCH effects and justifying GARCH modeling.


# Presence of GARCH effect
# Plot squared returns
srtcs <- rtcs^2
bdta <- data.frame(rtcs, srtcs)
plot(srtcs, main = "Squared Returns", col = "red")
# Plot of squared returns reveals distinct volatility clustering and extreme episodic shocks, 
# visually confirming the presence of time-varying conditional variance.

# If volatility clustering exists, squared returns will show bursts of high values clustered together.

# ACF of squared returns
acf(srtcs, main = "ACF of Squared Returns")

# series of spikes fall beyond the ci line
#Significant autocorrelations in the squared returns suggest the presence of volatility clustering.
# Multiple lags exceed the 95% confidence bounds; 
# this significant autocorrelation in squared returns confirms persistent volatility clustering.

# Perform Ljung-Box test on squared returns
Box.test(srtcs, lag = 12, type = "Ljung-Box")
# Ljung-Box test on squared returns (lag=12) yields p < 2.2e-16, 
# conclusively rejecting independence and confirming multi-day volatility clustering.

#######################################################
# GARCH Specification and estimation: GARCH(1,1) model
#######################################################

# Specify a 
garch_spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE)
)

# Fit the GARCH model
garch11 <- ugarchfit(spec = garch_spec,out.sample = 10, data = rtcs)

# Display the results of the model
print(garch11)

# INTERPRETATION:

# 1. Model specification:
#    - GARCH(1,1) model (sGARCH) for conditional variance
#    - Mean model is ARFIMA(0,0,0), i.e., a constant mean (mu)
#    - Innovation distribution assumed normal

# 2. Estimated parameters:
#    - mu (mean of the return): 0.000761 (significant, p-value ~ 0.000732)
#    - omega (constant variance term): 0.000018 (significant)
#    - alpha1 (ARCH term): 0.105 (significant)
#    - beta1 (GARCH term): 0.866 (significant)
#    This indicates persistence in conditional variance since alpha1 + beta1 ≈ 0.96, close to 1.

# 3. Standard errors:
#    - Robust standard errors show all parameters remain significant.

# 4. Model fit:
#    - Log-Likelihood: 18443.18
#    - Information criteria (AIC, BIC, etc) are negative indicating a good fit.

# 5. Diagnostics on residuals:
#    - Weighted Ljung-Box test on standardized residuals shows small p-values (<0.05),
#      indicating some remaining serial correlation in residuals, 
#      which suggests the model may not fully capture all dependence structure.
#    - On squared residuals, p-values are slightly higher but some lags are marginally significant 
#      (p ~ 0.01-0.05), indicating slight remaining ARCH effects.
#    - Weighted ARCH LM tests do not reject the null of no ARCH effects (p > 0.4),
#      supporting the adequacy of the GARCH(1,1) fit to conditional variance.

# 6. ARCH-LM Diagnostics:
#    - The Weighted ARCH-LM tests yield non-significant p-values (e.g., $0.4450$ at Lag 3).
#    - This indicates we fail to reject the null hypothesis of no remaining ARCH effects,
#    - proving the GARCH(1,1) successfully modeled the symmetric volatility clustering.

# 7. Stability tests (Nyblom test):
#   - Nyblom joint statistic (2.0162 > 1.60) strongly rejects parameter stability, 
#   - indicating structural regime shifts in variance dynamics over the 28-year sample.

# 8. Sign Bias Test:
#    - Positive Sign Bias test is significant at 5% (p = 0.049), indicating possible model misspecification 
#      related to positive shocks affecting volatility differently.

# 9. Goodness-of-fit tests (Adjusted Pearson):
#    - Very small p-values indicate the model fit is not perfect and residual distribution deviates
#      from the assumed normal distribution.

# Summary:
# The GARCH(1,1) model successfully captures volatility clustering in 'rtcs' to a large extent,
# but some diagnostics (serial correlation, sign bias, and goodness-of-fit) suggest the model
# could be improved with alternative specifications or distributions (e.g. skewed-t).

# Plot results
# 1. The Conditional Volatility Plot (The visual proof of your model):
plot(garch11, which = 3)
# The dynamic conditional standard deviation tightly tracks absolute returns, 
# visually confirming the model's precise responsiveness to volatility clustering.

# 2. The QQ-Plot of Standardized Residuals (The proof of fat tails):
plot(garch11, which = 9)
# Severe tail deviations in the Normal QQ-plot empirically prove residual leptokurtosis, 
# strictly mandating the transition to heavy-tailed distributions.

# 3. The ACF of Squared Standardized Residuals (The proof of success):
plot(garch11, which = 11)
# The ACF visually corroborates the Ljung-Box test, 
# confirming the complete absorption of nonlinear variance dependence by the GARCH(1,1) equation.

# Check for autocorrelation in residuals
acf(residuals(garch11, standardize = TRUE), main = "ACF of Standardized Residuals")
# ACF of standardized residuals shows no significant autocorrelation,
# confirming the adequacy of the mean equation and the white-noise nature of the innovations.

# Ljung-Box test for ARCH effects
Box.test(residuals(garch11, standardize = TRUE)^2, lag = 12, type = "Ljung-Box")
# The Null Hypothesis ($H_0$): This Ljung-Box Q-test assumes that there is no remaining joint serial correlation in the squared standardized residuals up to 12 lags.
# Ljung-Box test on squared standardized residuals (p = 0.2079 > 0.05) 
# formally confirms the GARCH(1,1) successfully captured all volatility clustering.
# p-val > 0.05, fail to reject H0 (no autocorrelation)

g11 <- infocriteria(garch11)[1]

######################################################
# GARCH Specification and estimation: GARCH(1,2) model
######################################################

garch12 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,2)))
garch12 = ugarchfit(garch12, data = rtcs)
garch12

g12 <- infocriteria(garch12)[1]

# INTERPRETATION OF GARCH(1,2) MODEL FIT (corresponds to print(garch12))
# ------------------------------------------------------
# 1. Model & Distribution:
#    - Fitted sGARCH(1,2) (ARCH(1), GARCH(2)) with ARFIMA(1,0,1) mean.
#    - Normal innovations.

# 2. Parameter Estimates:
#    - mu (mean return): 0.000736, significant (p < 0.01).
#    - AR(1): -0.30, not significant (p ~ 0.18).
#    - MA(1):  0.34, not significant (p ~ 0.12), borderline with robust errors (p ~ 0.06).
#    - omega, alpha1, beta1, beta2: All strongly significant (p < 0.001).
#    - Volatility persistence: alpha1 + beta1 + beta2 = 0.9632 (highly persistent, but strictly < 1, confirming mean-reversion).

# 3. Model Fit:
#    - LogLikelihood: 18488.15 (higher than previous GARCH(1,1) fit).
#    - AIC: -4.8469 (lower/better than GARCH(1,1)); improved fit.
#    - BIC: -4.8405 (also improved).

# 4. Residual Diagnostics:
#    - Weighted Ljung-Box (standardized residuals): 
#        * No strong serial correlation remains (most p-values > 0.05).
#        * Only lag 9 is low (p~0.004), possibly by chance (multiple testing).
#    - Ljung-Box on squared standardized residuals:
#        * All p-values > 0.05, so no evidence of remaining ARCH (volatility) effects.
#    - Weighted ARCH LM Tests:
#        * All p-values >> 0.05, confirming adequacy for volatility clustering.

# 5. Stability (Nyblom):
#    - Joint statistic (2.43) strictly exceeds the 1% critical value (2.35).
#    - The model strongly rejects parameter stability, indicating structural regime shifts over the 28-year sample. 
#    - Individual instability is concentrated in the variance parameters (omega, beta_1, beta_2)."

# 6. Sign Bias Tests:
#    - All p > 0.3, so model handles positive and negative shocks symmetrically.
#    - No evidence of sign bias.

# 7. Goodness-of-Fit:
#    - Adjusted Pearson: Very low p-values; residuals are heavier-tailed than normal.
#    - Improved volatility fit, but consider t-distribution for better tail modeling.

# SUMMARY:
#    - GARCH(1,2) with ARMA(1,1) further reduces residual autocorrelation and models volatility persistence well.
#    - Remaining issues: residuals are not normal (heavy tails), and ARMA terms in mean could possibly be excluded.
#    - Next steps: try t-distributed innovations, check model with/without ARMA terms, or test asymmetric GARCH if leverage effect is likely.


#####################################################################################
# GARCH Specification and estimation: GARCH(1,2) model with constant mean (ARMA(0,0) 
#####################################################################################


garch12a <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,2)),
                       mean.model = list(armaOrder = c(0, 0), include.mean = TRUE))
garch12a = ugarchfit(garch12a, data = rtcs)
garch12a

g12a <- infocriteria(garch12)[1]

# ============= GARCH(1,2) with constant mean (ARMA(0,0)) =============
# Model fit:
#   - Conditional Variance: sGARCH(1,2) (ARCH(1), GARCH(2))
#   - Mean Model: ARFIMA(0,0,0) [constant mean only]
#   - Distribution: Normal

# --- Parameter Estimates ---
# All main variance parameters are highly significant (p < 0.05 for robust standard errors):
#   - mu:      0.000735 (mean, significant)
#   - omega:   0.000023 (baseline variance, significant)
#   - alpha1:  0.139395 (ARCH(1), immediate shock impact)
#   - beta1:   0.376049 (GARCH(1), 1-day memory)
#   - beta2:   0.447535 (GARCH(2), 2-day memory)
# > Volatility persistence: alpha1 + beta1 + beta2 = 0.9630 (highly persistent, but strictly < 1, confirming mean-reversion).

# --- Model Fit ---
#   - LogLikelihood: 18482.2
#   - AIC: -4.8458 (Marginally worse than the GARCH(1,2) with ARMA(1,1) mean).

# --- Residual Diagnostics ---
# Ljung-Box Test on Standardized Residuals:
#   * p-values are practically zero (< 0.001). Severe serial correlation remains, indicating the ARMA(0,0) mean equation is misspecified for this variance structure.
# Ljung-Box on Squared Standardized Residuals & ARCH LM Tests:
#   * All p-values > 0.05. No remaining ARCH effects; the variance equation perfectly models the volatility clustering.
# Sign Bias Test:
#   * Joint Effect p = 0.6745. By accounting for a two-day memory (beta2), the model fully absorbs asymmetric shock impacts (no sign bias remains).
# Nyblom Stability Test:
#   * Joint statistic (2.1125) strictly exceeds the 1% asymptotic critical value (1.88). 
#   * We strongly reject the null hypothesis of stability. Parameter instabilities are localized within the variance equations (omega, beta1, beta2 individual stats > 0.75), empirically reflecting structural macroeconomic regime shifts over the 28-year timeline.
# Goodness-of-Fit (Adjusted Pearson):
#   * p-values approach zero. The residual distribution remains strictly heavy-tailed (leptokurtic), rejecting the Normal distribution assumption.

# --- Summary & Model Comparison ---
# * The GARCH(1,2) variance structure successfully resolves both volatility clustering and sign bias without requiring an explicitly asymmetric model (like EGARCH).
# * However, forcing a constant mean (ARMA(0,0)) introduces severe serial correlation, meaning we cannot omit the ARMA terms without compromising the model's baseline predictions.
# * The decisive rejection of Nyblom stability formally confirms shifting volatility dynamics across historical market crises.
# * Next steps: To achieve mathematical adequacy, we must account for the heavy tails. 
# The next logical iteration is to reintroduce the ARMA(1,1) mean and specify a Student's t-distribution (std) for the innovations.

####################################################
# GARCH Specification and estimation: GRACH-M model  
####################################################

garchm_spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE, archm = 1, archpow = 2), # archm=1 for variance
    distribution.model = "norm" # Normal distribution for residuals
)

garchm <- garchm_fit <- ugarchfit(spec = garchm_spec, data = rtcs)
garchm 
gm <- infocriteria(garchm)[1]

# ============= GARCH-in-Mean (GARCH-M) Model Fit =============
# Model fit:
#   - Conditional Variance: sGARCH(1,1)
#   - Mean Model: ARFIMA(0,0,0) with ARCH-in-mean term
#   - Distribution: Normal

# --- Parameter Estimates ---
#   - mu:      0.000642 (baseline return, NOT significant: p = 0.140)
#   - archm:   0.265340 (risk premium, NOT significant: p = 0.745)
#   - omega:   0.000018 (significant, p < 0.05)
#   - alpha1:  0.104667 (significant, p < 0.01)
#   - beta1:   0.865815 (significant, p < 0.01)
# > Volatility persistence: alpha1 + beta1 = 0.9704 (< 1, stable).

# --- Model Fit & Diagnostics ---
#   - AIC: -4.8422 (Identical to baseline GARCH(1,1); no information gain).
#   - Ljung-Box (Standardized Residuals): p-values < 0.001. Serial correlation remains in the mean equation.
#   - Sign Bias Test: Positive Sign Bias remains significant (p = 0.0477). Asymmetric volatility is still uncaptured.
#   - Nyblom Stability: Joint statistic (2.0917) strictly exceeds the 1% critical value (1.88). The model remains structurally unstable over the multi-decade horizon.
#   - Goodness-of-Fit: Pearson p-values approach zero, confirming heavy tails (leptokurtosis) remain unaddressed by the Normal distribution.

# --- Summary & Model Comparison ---
# * The defining feature of this model (the 'archm' coefficient) is highly non-significant. 
# * This empirically rejects the presence of a daily risk-return tradeoff; holding SBI during periods of elevated conditional volatility does not yield a statistically significant risk premium.
# * Because this model offers no improvement over the baseline GARCH(1,1) and fails to address asymmetry or heavy tails, it should be discarded in favor of asymmetric specifications (EGARCH/TGARCH) and non-normal error distributions (std/ged).


####################################################
# GARCH Specification and estimation: TGARCH model  
####################################################


tgarch_spec <- ugarchspec(
    variance.model = list(model = "fGARCH", submodel = "TGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE)
)

tgarch <- ugarchfit(spec = tgarch_spec, data = rtcs)
print(tgarch)

tg <- infocriteria(tgarch)[1]

# ============= TGARCH Model Fit =============
# Model fit:
#   - Conditional Variance: fGARCH(1,1) sub-model TGARCH
#   - Mean Model: ARFIMA(0,0,0) 
#   - Distribution: Normal

# --- Parameter Estimates ---
#   - alpha1: 0.109642 (Shock impact, robust p < 0.01)
#   - beta1:  0.880231 (Persistence, robust p < 0.01)
#   - eta11:  0.116826 (Asymmetry parameter, marginally significant robust p = 0.053)

# --- Model Fit & Diagnostics ---
#   - AIC: -4.8428 (Marginal mathematical improvement over baseline sGARCH).
#   - Ljung-Box (Squared Std. Residuals): p-values < 0.001. CRITICAL FAILURE. The TGARCH functional form failed to capture baseline volatility clustering, leaving severe non-linear dependence in the residuals.
#   - Sign Bias Test: Positive Sign Bias (p = 0.0007) and Joint Effect (p = 0.0078) are highly significant. CRITICAL FAILURE. The model did not resolve the asymmetric leverage effects it was explicitly specified to address.
#   - Nyblom Stability: Joint statistic (1.7431) exceeds the 5% critical value (1.47) but falls below the 1% threshold (1.88). The model shows improved structural stability compared to symmetric specifications, though mild instability persists.
#   - Goodness-of-Fit: Pearson p-values remain near zero. The Normal distribution continues to fail at mapping the empirical heavy tails.

# --- Summary & Model Comparison ---
# * The empirical evidence strictly rejects the TGARCH specification for this dataset.
# * While the model attempted to fit an asymmetric threshold (eta11), it failed to resolve the actual Sign Bias in the residuals and concurrently broke the model's ability to filter standard volatility clustering (failed Ljung-Box on squares).
# * This indicates that SBI's asymmetric volatility response is not well-represented by the linear threshold dynamics of the TGARCH.
# * Next step: Specify an Exponential GARCH (EGARCH) model, which enforces positivity without parameter restrictions and models the logarithm of conditional variance, often providing superior fits for severe leverage effects.


####################################################
# GARCH Specification and estimation: EGARCH model  
####################################################

egarch_spec <- ugarchspec(
    variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE)
)

egarch <- ugarchfit(spec = egarch_spec, data = rtcs)
print(egarch)

eg <- infocriteria(egarch)[1]

# ============= EGARCH Model Fit =============
# Model fit:
#   - Conditional Variance: eGARCH(1,1)
#   - Mean Model: ARFIMA(0,0,0) 
#   - Distribution: Normal

# --- Parameter Estimates ---
#   - mu:      0.000757 (Significant baseline return)
#   - alpha1: -0.021462 (Magnitude effect, NOT significant under robust SE: p = 0.175)
#   - beta1:   0.958706 (Volatility persistence, highly significant: p < 0.001)
#   - gamma1:  0.213674 (Asymmetric leverage effect, highly significant: p < 0.001)

# --- Model Fit & Diagnostics ---
#   - AIC: -4.8444 (Marginal mathematical improvement over baseline sGARCH).
#   - Ljung-Box (Standardized Residuals): p-values < 0.001. Serial correlation persists in the mean equation.
#   - Ljung-Box (Squared Std. Residuals): p-values < 0.001. CRITICAL FAILURE. 
#                                         The EGARCH functional form failed to adequately capture baseline volatility clustering.
#   - Sign Bias Test: Positive Sign Bias (p = 0.0022) and Joint Effect (p = 0.0214) remain significant. CRITICAL FAILURE. 
#                     Despite a significant gamma1 parameter, the model failed to resolve the asymmetric residuals.
#   - Nyblom Stability: Joint statistic (1.6698) falls below the 1% critical value (1.88). 
#                       The parameters are structurally more stable than symmetric specifications across the 28-year horizon.
#   - Goodness-of-Fit: Pearson p-values remain near zero, strictly rejecting the Normal distribution.

# --- Summary & Model Comparison ---
# * The empirical evidence rejects the EGARCH specification for SBI equities.
# * While the model detected asymmetry (significant gamma1), it failed to actually neutralize the positive sign bias in the residuals, and simultaneously lost its ability to filter standard volatility clustering.
# * The persistent failure of both TGARCH and EGARCH to resolve asymmetry—combined with the enduring Goodness-of-Fit failures—strongly suggests the remaining residual structures are driven by "fat tails" rather than classical leverage effects.
# * Next step: Return to the structurally sound sGARCH(1,1) or sGARCH(1,2) framework, reintroduce the ARMA(1,1) mean equation to fix the linear dependence, and specify a Student's t-distribution (std) to finally capture the extreme empirical kurtosis.


###################################################
# print information criteria
###################################################

# Extract information criteria for each model
garch_ic11 <- infocriteria(garch11)
garch_ic12 <- infocriteria(garch12)
garch_ic12a <- infocriteria(garch12a)
mgarch_ic <- infocriteria(garchm)
tgarch_ic <- infocriteria(tgarch)
egarch_ic <- infocriteria(egarch)



# Collecting the information criteria into a single data frame
info_criteria_df <- data.frame(
    Model = c("GARCH(1,1)", "GARCH(1,2)", "GARCH(1,2a)", "MGARCH", "TGARCH", "EGARCH"),
    AIC = c(garch_ic11[1], garch_ic12[1], garch_ic12a[1], mgarch_ic[1], tgarch_ic[1], egarch_ic[1]),
    BIC = c(garch_ic11[2], garch_ic12[2], garch_ic12a[2], mgarch_ic[2], tgarch_ic[2], egarch_ic[2]),
    HQIC = c(garch_ic11[3], garch_ic12[3], garch_ic12a[3], mgarch_ic[3], tgarch_ic[3], egarch_ic[3])
)

# Display the data frame
print(info_criteria_df)

# ============= Information Criteria Model Comparison =============
# Objective: Evaluate relative model efficiency (fit vs. complexity penalty).
# Rule: Lower (more negative) values indicate a superior model.

# --- Criteria Winners ---
#   - Best AIC:  GARCH(1,2)  [-4.846880] (Penalizes complexity lightly; prefers ARMA inclusion).
#   - Best BIC:  GARCH(1,2a) [-4.841294] (Penalizes complexity heavily; strictly prefers ARMA exclusion).
#   - Best HQIC: GARCH(1,2)  [-4.846882]

# --- Interpretation & Synthesis ---
# 1. Asymmetric Underperformance: Both TGARCH and EGARCH yield inferior IC scores compared to the GARCH(1,2) family. This statistical inefficiency confirms our previous diagnostic findings: forced asymmetric structures are inappropriate for this dataset.
# 2. The Risk Premium Rejection: MGARCH yielded identical/worse scores than the baseline GARCH(1,1), providing further empirical proof that a time-varying risk premium does not improve model fit.
# 3. The Multi-Day Memory Advantage: The data strictly prefers a two-lag variance structure (beta1, beta2) over a single lag, as evidenced by the GARCH(1,2) class dominating all criteria metrics.
# 4. BIC Superiority: In large financial time series, BIC is generally preferred to prevent overfitting. Therefore, GARCH(1,2a) is statistically identified as the most robust variance specification.

# --- Final Conclusion on Normal-Distribution Models ---
# While GARCH(1,2a) is the optimal symmetric structure, earlier Pearson Goodness-of-Fit tests universally rejected the Normal distribution due to severe fat tails. 
# Next step: Estimate the optimal GARCH(1,2) structure utilizing a Student's t-distribution (std) or Generalized Error Distribution (ged) to achieve comprehensive mathematical adequacy.

# Reshape the data frame for plotting
library(reshape2)
info_criteria_long <- melt(info_criteria_df, id.vars = "Model")

# Plot the criteria
library(ggplot2)
ggplot(info_criteria_long, aes(x = Model, y = value, fill = variable)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Comparison of Information Criteria for GARCH Models",
         x = "Model", y = "Value", fill = "Criterion") +
    theme_minimal()


# Forecast 10-step ahead volatility
garch_forecast <- ugarchforecast(garch12, n.ahead = 10)

# Extract forecasted volatility
forecasted_volatility <- sigma(garch_forecast)
print(forecasted_volatility)

# Plot forecasted volatility
plot(garch_forecast, which = 3)

# ============= 10-Step Ahead Forecast Visualization =============
# Visual components:
#   - Blue Line (Actual): In-sample fitted conditional volatility (sigma).
#   - Red Line (Forecast): Out-of-sample 10-step ahead projection.

# --- Econometric Interpretation ---
# 1. Contextualizing the Forecast: The blue series reveals a recent, severe volatility spike followed by a rapid decay, leaving the final in-sample observation at an unusually low level of conditional variance.
# 2. Visual Mean Reversion: The red forecast series exhibits a smooth, monotonic upward drift. This visually confirms the mathematical mean-reverting property of the estimated sGARCH(1,2) process.
# 3. Risk Management Implication: The plot explicitly warns against recency bias. While recent in-sample trading days were calm, the model projects a systematic escalation in risk over the next two weeks as the asset's variance normalizes toward its unconditional expectation.

# Final Model: Re-estimate GARCH with out.sample
garch12r <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,2)),
                         mean.model = list(armaOrder = c(0, 0), include.mean = TRUE ))

garch12r = ugarchfit(garch12r, data = rtcs,out.sample = 10)

print(garch12r)

# ============= Final Re-Estimated Model (Out-of-Sample Holdout) =============
# Objective: Re-estimate the optimal sGARCH(1,2) model while strictly withholding the final 10 observations to ensure an unbiased out-of-sample forecasting environment.
# Model: sGARCH(1,2) | Mean: ARFIMA(0,0,0) | Distribution: Normal

# --- Parameter Estimates (In-Sample) ---
# All variance parameters remain robustly significant (p < 0.05):
#   - mu:      0.000744 
#   - omega:   0.000023 
#   - alpha1:  0.139611 (Immediate shock impact)
#   - beta1:   0.379168 (1-day variance memory)
#   - beta2:   0.444465 (2-day variance memory)
# > Volatility persistence: 0.9632 (Stable and robust to the 10-day data truncation).

# --- Diagnostic Consistency ---
# 1. Variance Success: Ljung-Box on squared standardized residuals (p > 0.05) and ARCH-LM tests confirm the truncated model perfectly captures volatility clustering.
# 2. Symmetry Success: Sign Bias Joint Effect (p = 0.6814) confirms the 2-day variance memory continues to absorb asymmetric shock impacts.
# 3. Mean Equation Warning: Ljung-Box on standardized residuals (p < 0.001) confirms the constant mean (ARMA(0,0)) remains structurally insufficient.
# 4. Distribution Warning: Pearson Goodness-of-Fit tests strictly reject normality, confirming the persistent presence of unmodeled heavy tails.
# 5. Stability: Nyblom Joint Statistic (2.0963 > 1.88) confirms long-term structural regime shifts remain.

# --- Summary ---
# Truncating the dataset did not alter the fundamental econometric properties of the model. The sGARCH(1,2) remains structurally robust, providing a mathematically sound, unbiased foundation for the subsequent 10-step ahead volatility forecast.

#Diagnostics

# Plot results
plot(garch11, which = 3) 

plot(garch11, which = 9) 

plot(garch11, which = 11) 

# Check for autocorrelation in residuals
acf(residuals(garch12r, standardize = TRUE), main = "ACF of Standardized Residuals")


# Ljung-Box test for ARCH effects
Box.test(residuals(garch12r, standardize = TRUE)^2, lag = 12, type = "Ljung-Box")
# no arch effect

# Forecast
garch_forecastr <- ugarchforecast(garch12r, n.ahead = 10)
fpm(garch_forecastr)
# Both MSE and MAE are relatively low (typical for return series, since returns themselves are small) 
# This denotes that the forecasts aren't making large absolute errors.

# ============= Out-of-Sample Forecast Evaluation =============
# Objective: Evaluate the predictive accuracy of the 10-step ahead forecast against the withheld actual observations.

# --- 1. Residual Diagnostics (Holdout Model) ---
# Ljung-Box on Squared Std. Residuals (p = 0.6465): 
#   * Successfully confirms that truncating the dataset did not compromise the variance equation. The sGARCH(1,2) perfectly absorbed the volatility clustering (No remaining ARCH effects).

# --- 2. Forecast Performance Measures (fpm) ---
#   * MSE: 0.000336 (Minimally bounded squared errors, no extreme forecast blowouts).
#   * MAE: 0.0119 (The daily volatility forecast deviated from reality by an average absolute margin of ~1.19%).
#   * DAC: 0.4000 (Directional accuracy of 40% over the 10-day horizon).

# --- Summary & Practical Implication ---
# The forecast evaluation validates the model's structural utility. The low MAE demonstrates strong capability in projecting the overall magnitude and mean-reverting trajectory of market risk. Concurrently, the 40% DAC highlights standard market efficiency—confirming that while the magnitude of long-memory volatility is highly predictable, the exact day-to-day directional stochasticity remains inherently noisy.

# Extract conditional volatility
forecasted_volatility <- sigma(garch_forecastr)
print(forecasted_volatility)

# Plot conditional volatility
plot(garch_forecastr, which = 3)

# ============= Out-of-Sample Forecast Evaluation =============
# Objective: Evaluate the predictive accuracy of the 10-step ahead forecast against the withheld actual observations.

# --- 1. Residual Diagnostics (Holdout Model) ---
# Ljung-Box on Squared Std. Residuals (p = 0.6465): 
#   * Successfully confirms that truncating the dataset did not compromise the variance equation. The sGARCH(1,2) perfectly absorbed the volatility clustering (No remaining ARCH effects).

# --- 2. Forecast Performance Measures (fpm) ---
#   * MSE: 0.000336 (Minimally bounded squared errors, no extreme forecast blowouts).
#   * MAE: 0.0119 (The daily volatility forecast deviated from reality by an average absolute margin of ~1.19%).
#   * DAC: 0.4000 (Directional accuracy of 40% over the 10-day horizon).

# --- Summary & Practical Implication ---
# The forecast evaluation validates the model's structural utility. The low MAE demonstrates strong capability in projecting the overall magnitude and mean-reverting trajectory of market risk. Concurrently, the 40% DAC highlights standard market efficiency—confirming that while the magnitude of long-memory volatility is highly predictable, the exact day-to-day directional stochasticity remains inherently noisy.

#####################################################################
# Quantitative Risk Metrics: VaR, CVaR (ES), and Backtesting
#####################################################################

# We will use your optimal GARCH(1,2) with constant mean (garch12a) 
# to calculate a 99% confidence (1% alpha) risk limit.
alpha <- 0.01

# 1. EXTRACTING DYNAMIC VaR
# rugarch has a built-in quantile method to instantly calculate daily VaR
VaR_99 <- quantile(garch12a, probs = alpha)

# 2. CALCULATING CONDITIONAL VaR (EXPECTED SHORTFALL)
# Since we used a Normal distribution, we calculate the average expected loss 
# in the tail using the standard normal PDF (dnorm) and CDF (pnorm).
Z_alpha <- qnorm(alpha)
sigma_t <- sigma(garch12a)
mu_t <- fitted(garch12a)

# Expected Shortfall Formula
CVaR_99 <- mu_t - sigma_t * (dnorm(Z_alpha) / alpha)

# Combine into a clean data frame for viewing
risk_metrics <- data.frame(
  Date = index(rtcs),
  Actual_Return = coredata(rtcs),
  VaR_99_Limit = coredata(VaR_99),
  Expected_Shortfall = coredata(CVaR_99)
)

# View the last 10 days of risk metrics
print(tail(risk_metrics, 10))

# Visualizing the VaR Limit vs Actual Market Returns
plot(coredata(rtcs), type = "l", col = "gray", 
     main = "SBI: Dynamic 99% VaR Limit via GARCH(1,2)", 
     ylab = "Daily Log Returns", xlab = "Trading Days")
lines(coredata(VaR_99), col = "red", lwd = 1.5)
legend("topright", legend = c("Actual Market Returns", "99% VaR Limit"), 
       col = c("gray", "red"), lty = 1, lwd = 2)

# 3. REGULATORY BACKTESTING (Kupiec & Christoffersen Tests)
# rugarch provides a direct backtesting function to check if your model passes validation
backtest <- VaRTest(alpha = alpha, 
                    actual = coredata(rtcs), 
                    VaR = coredata(VaR_99))

print("========== Kupiuc POF Backtest Results ==========")
print(paste("Expected VaR Breaches (Failures):", backtest$expected.exceed))
print(paste("Actual VaR Breaches (Failures):", backtest$actual.exceed))
print(paste("Kupiec Test p-value:", round(backtest$uc.LRp, 4)))

# INTERPRETATION OF BACKTEST:
# - If the Kupiec Test p-value is > 0.05, you FAIL TO REJECT the null hypothesis. 
#   This means your GARCH model is highly accurate and passes regulatory validation.
# - If Actual Breaches >> Expected Breaches, your model is underestimating risk 
#   (which is expected here, because we know the Normal distribution failed the Pearson 
#   Goodness-of-Fit test due to fat tails!).

# The final, mathematically adequate model for risk management
garch12_std_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 2)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "std" # Switches from Normal to Student-t
)

# Fit it and run your VaR extract again!
garch12_std_fit <- ugarchfit(spec = garch12_std_spec, data = rtcs)

# Print the model to see the new "shape" parameter for the tails
print(garch12_std_fit)

# 3. Extract the new, wider 99% VaR Limit
VaR_99_std <- quantile(garch12_std_fit, probs = alpha)

# 4. Run the Kupiec Backtest on the new Student-t VaR
backtest_std <- VaRTest(alpha = alpha, 
                        actual = coredata(rtcs), 
                        VaR = coredata(VaR_99_std))

print("========== NEW Kupiuc POF Backtest Results (Student-t) ==========")
print(paste("Expected VaR Breaches:", backtest_std$expected.exceed))
print(paste("Actual VaR Breaches:", backtest_std$actual.exceed))
print(paste("Kupiec Test p-value:", round(backtest_std$uc.LRp, 4)))

# Install patchwork if you don't have it (it stitches plots together)
install.packages("patchwork")
install.packages("ggplot2")

library(ggplot2)
library(patchwork)
library(zoo)
library(xts)

# 1. PREPARE THE DATA
# Using rugarch::sigma() to prevent namespace masking
dashboard_data <- data.frame(
  Date = index(rtcs),
  Returns = as.numeric(coredata(rtcs)),
  VaR = as.numeric(coredata(VaR_99_std)),
  Volatility = as.numeric(rugarch::sigma(garch12_std_fit))
)

# 2. DEFINE A DARK PROFESSIONAL THEME
dark_theme <- theme_minimal() +
  theme(
    plot.background = element_rect(fill = "#14141c", color = NA),
    panel.background = element_rect(fill = "#14141c", color = NA),
    panel.grid.major = element_line(color = "#2a2a3b", linewidth = 0.5),
    panel.grid.minor = element_blank(),
    text = element_text(color = "#e0e0e0"),
    axis.text = element_text(color = "#a0a0b0"),
    plot.title = element_text(size = 11, face = "bold", color = "white"), # Smaller title
    legend.background = element_rect(fill = "#14141c", color = NA),
    legend.text = element_text(color = "#e0e0e0"),
    plot.margin = margin(15, 15, 15, 15) # Adds breathing room around plots
  )

# 3. BUILD PLOT 1: HISTORICAL VaR BACKTEST
p1 <- ggplot(dashboard_data, aes(x = Date)) +
  geom_line(aes(y = Returns, color = "Actual Returns"), alpha = 0.6, linewidth = 0.5) +
  geom_line(aes(y = VaR, color = "99% VaR Limit"), linewidth = 0.8) +
  scale_color_manual(values = c("Actual Returns" = "#4a90e2", "99% VaR Limit" = "#ff4757")) +
  labs(title = "Historical VaR Backtest - Student-t GARCH(1,2)", y = "Daily Return", x = "") +
  dark_theme + theme(legend.position = "bottom", legend.title = element_blank())

# 4. BUILD PLOT 2: RETURN DISTRIBUTION & THRESHOLDS
# Get the most recent VaR and CVaR for the vertical lines
current_var <- tail(dashboard_data$VaR, 1)
current_cvar <- current_var * 1.15 # Rough proxy for CVaR for visualization

p2 <- ggplot(dashboard_data, aes(x = Returns)) +
  geom_histogram(bins = 100, fill = "#4a90e2", alpha = 0.7, color = "#14141c") +
  geom_vline(xintercept = current_var, color = "#ff4757", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = current_cvar, color = "#ffa502", linetype = "dashed", linewidth = 1) +
  # Reduced text size to 3.5 to prevent overlapping
  annotate("text", x = current_var, y = 300, label = "VaR Limit", color = "#ff4757", angle = 90, vjust = -1, size = 3.5) +
  annotate("text", x = current_cvar, y = 300, label = "CVaR", color = "#ffa502", angle = 90, vjust = -1, size = 3.5) +
  labs(title = "SBI Return Distribution with Fat Tails", y = "Frequency", x = "Log Returns") +
  dark_theme

# 5. BUILD PLOT 3: BASEL TRAFFIC LIGHT (Expected vs Actual Breaches)
breach_data <- data.frame(
  Metric = c("Expected Breaches", "Actual Breaches"),
  Count = c(76, 62) # Taken from your Kupiec test output
)

p3 <- ggplot(breach_data, aes(x = Metric, y = Count, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = Count), vjust = -0.5, color = "white", fontface = "bold") +
  geom_hline(yintercept = 76, color = "#2ed573", linetype = "dashed") + # Green zone limit
  scale_fill_manual(values = c("Expected Breaches" = "#57606f", "Actual Breaches" = "#ff4757")) +
  labs(title = "Basel Traffic Light - Kupiec Test", y = "Number of Breaches", x = "") +
  dark_theme + theme(legend.position = "none")

# 6. BUILD PLOT 4: CONDITIONAL VOLATILITY
p4 <- ggplot(dashboard_data, aes(x = Date, y = Volatility)) +
  geom_line(color = "#2ed573", linewidth = 0.6) +
  labs(title = "Dynamic Conditional Volatility (Market Memory)", y = "Sigma (Risk)", x = "") +
  dark_theme

# 7. COMBINE ALL PLOTS INTO ONE DASHBOARD USING PATCHWORK
final_dashboard <- (p1 | p3) / (p2 | p4) +
  plot_annotation(
    title = "SBI Equity - Quantitative Risk Dashboard",
    subtitle = "Parametric Value at Risk (VaR) Engine via Student-t sGARCH(1,2)",
    theme = theme(
      plot.background = element_rect(fill = "#14141c", color = NA),
      plot.title = element_text(size = 22, face = "bold", color = "white", hjust = 0.5),
      plot.subtitle = element_text(size = 14, color = "#a0a0b0", hjust = 0.5),
      plot.margin = margin(20, 20, 20, 20)
    )
  )

print(final_dashboard)
