---
layout: notes
section-type: notes
title: time series
category: stat
---
* TOC
{:toc}
---

# high-level

## basics

- usually assume points are equally spaced
- modeling - for understanding underlying process or predicting
- [nice blog](https://algorithmia.com/blog/introduction-to-time-series), [nice tutorial](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc442.htm), [Time Series for scikit-learn People](https://www.ethanrosenthal.com/2018/01/28/time-series-for-scikit-learn-people-part1/)
- *noise*, *seasonality* (regular / predictable fluctuations), *trend*, *cycle*
- multiplicative models: time series = trend * seasonality * noise
- additive model: time series = trend + seasonality + noise
- stationarity - mean, variance, and autocorrelation structure do not change over time
- exogenous variable = y = dependent variable = value is determined outside the model and is imposed on the model
- endogenous variable = x = independent variable = regressor - value is determined by model

## libraries

- [pandas has some great time-series functionality](https://tomaugspurger.github.io/modern-7-timeseries)
- [skits library](https://github.com/EthanRosenthal/skits) for forecasting

## high-level modelling

- common methods
  - decomposition - identify each of these components given a time-series
    - ex. loess, exponential smoothing
  - frequency-based methods - e.g. look at spectral plot
  - (AR) autoregressive models - linear regression of current value of one or more prior values of the series
  - (MA) moving-average models - require fitting the noise terms
  - (ARMA) box-jenkins approach 
- moving averages
  - simple moving average - just average over a window
  - cumulative moving average - mean is calculated using previous mean
  - exponential moving average - exponentially weights up more recent points
- prediction (forecasting) models
  - autoregressive integrated moving average (arima)
    - assumptions: stationary model

## models

**AR model** $AR(p)$: $$ X_t = c + \sum_{i=1}^p \varphi_i X_{t-i}+ \varepsilon_t $$

- $\varphi_1, \ldots, \varphi_p$ are parameters
- $c$ is a constant
- $\varepsilon_t$ is white noise
- stationary assumption places constraints on param values (e.g. processes in the AR(1) model with $|\varphi_1| \ge 1$ are not stationary)

**MA model** $MA(q)$: $ X_t = \mu + \varepsilon_t + \sum_{i=1}^q \theta_i \varepsilon_{t-i}$

- $\theta_1 ... \theta_q$ are params
- $\mu$ is the mean of $X_t$ 
- $\varepsilon_t$, $\varepsilon_{t-1}$ are white noise error terms
- harder to fit, because the lagged error terms are not visible (also means can't make preds on new time-series)

**ARMA model**: $ARMA(p, q)$: $X_t = c + \varepsilon_t +  \sum_{i=1}^p \varphi_i X_{t-i} + \sum_{i=1}^q \theta_i \varepsilon_{t-i}$

**ARIMA model**: $ARIMA(p, d, q)$: - generalizes ARMA model to non-stationarity (using differencing)

- d number of nonseasonal differences (differencing order i.e. what order of derivative to take)



# [book](https://www.stat.tamu.edu/~suhasini/teaching673/time_series.pdf) notes

## ch 1

- usually in time-series analysis, we begin by de-trending the data and analyzing the residuals
  - ex. assume linear trend or quadratic trend and subtract that fit (or could include sin / cos for seasonal behavior)
  - ex. look at the differences instead of the points (nth order difference removes nth order polynomial trend). However, taking differences can introduce dependencies in the data
  - ex. remove trend using sliding window (maybe with exponential weighting)
- when errors are dependent, very hard to distinguish noise from signal
- periodogram - in FFT, this looks at the magnitude of the coefficients (but loses the phase information)

## ch 2 - stationary time series

- in time series, we never get iid data
- instead we make assumptions
  - ex. the process has a constant mean (a type of stationarity)
  - ex. the dependencies in the time-series are short-term
- autocorrelation plots: plot correlation of series vs series offset by different lags
- formal definitions of stationarity for time series $\{X_t\}$
  - **strict stationarity** - the distribution is the same across time
  - **second-order / weak stationarity** -  mean is constant for all t and if for any t and k the covariance between $X_t$ and $X_{t+k}$ only depends on the lag difference k
    - In other words, there exists a function $c: \mathbb Z \to \mathbb R$ such that for all t and k we have $c(k) = cov (X_t, X_{t+k})$
    - strict stationary and $E|X_T^2| < \infty \implies$ second-order stationary
  - **ergodic** - stronger condition, says samples approach the expectation of functions on the time series: for any function $g$ and shift $\tau_1, ... \tau_k$:
    - $\frac 1 n \sum_t g(X_t, ... X_{t+\tau_k}) \to \mathbb E [g(X_0, ..., X_{t+\tau_k} )]$
- **causal** - can predict given only past values (for Gaussian processes no difference)

## ch 3 - linear time series

- $AR(p)$ model: $X_t = \sum_{j=1}^p \phi_j X_{t-j} + \epsilon_t$
  - looks just like linear regression, but is more complex
    - if we don't account for issues, model will not be stationary, model may be misspecified, and $E(\epsilon_t|X_{t-p}) \neq 0$
    - this represents a set of difference equations, and as such, must have a solution
  - ex. $AR(1)$ model - if $|\phi|$ < 0, then soln is in terms of past values of {$\epsilon_t$}, otherwise it is in terms of future values
    - ex. simulating - if we know $\phi$ and $\{\epsilon_t\}$, we wstill need to use the backshift operator to solve for  $\{ X_t \}$
  - ex. $AR(p)$ model - if $\sum_j |\phi_j|$< 1, and $\mathbb E |\epsilon_t| < \infty$, then will have a causal stationary solution
  - **backshift operator** $B^kX_t=X_{t-k}$
    - solving requires using the backshift operator, because we need to solve for what all the residuals are
  - **characteristic polynomial** $\phi(B) = 1 - \sum_{j=1}^p \phi_j B^j$
    - $\phi(B) X_t = \epsilon_t$
    - $X_t=\phi(B)^{-1} \epsilon_t$
  - can represent $AR(p)$ as a vector $AR(1)$ using the vector $\bar X_t = (X_t, ..., X_{t-p+1})$
  - note: can reparametrize in terms of frequencies
- $MA(q)$ model: $X_t = \sum_{j=0}^q \theta_j \epsilon_{t-j}$ 
  - $E[\epsilon_t] = 0$, $Var[\epsilon_t] = 1$
  - much harder to estimate these parameters
  - $X_t = \theta (B) \epsilon_t$ (assuming $\theta_0=1$)
- $ARMA(p, q)$ - there are conditions for being invertible/causal + identifiable
- linear time-series = linear process - like MA$(\infty)$, but can depend on future observations as well
- $ARIMA(p, q)$ - just take differences first

## ch 4 - the autocovariance function

- autocovariance function: {$c(k): k \in \mathbb Z$} where $c(k) = \mathbb E (X_0 X_k)$
- **Yule-Walker equations** (assuming AR(p) process): $\mathbb E (X_t X_{t-k}) = \sum_{j=1}^p \phi_j \mathbb E (X_{t-j} X_{t-k}) + \underbrace{\mathbb E (\epsilon_tX_{t-k})}_{=0} = \sum_{j=1}^p \phi_j \mathbb E (X_{t-j} X_{t-k})$
- ex. MA covariance becomes 0 with lag > num params

## ch 8 - parameter estimation

- can rewrite the Yule-Walker equations:

  - $c(i) = \sum_{j=1}^p \phi_j c(i -j)$
  - $\underline r_p = \Sigma_p \underline \phi_p$
    - $(\Sigma_p)_{i, j} = c(i - j)$
  - $(\underline r_p)_k = c(i)$
    - $\underline \phi_p = (\phi_1, ..., \phi_p)$
    - this minimizes the mse $\mathbb E [X_{t+1} - \sum_{j=1}^p \phi_j X_{t+1-j}]^2$
  
-  use estimates to solve: $\hat{\underline \phi}_p = \hat \Sigma_p^{-1} \hat{\underline r}_p $

  

