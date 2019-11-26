---
layout: notes
section-type: notes
title: time series
category: stat
---
* TOC
{:toc}
---

# basics

- usually assume points are equally spaced
- modeling - for understanding underlying process or predicting
- [nice blog](https://algorithmia.com/blog/introduction-to-time-series), [nice tutorial](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc442.htm), [Time Series for scikit-learn People](https://www.ethanrosenthal.com/2018/01/28/time-series-for-scikit-learn-people-part1/)
- *noise*, *seasonality* (regular / predictable fluctuations), *trend*, *cycle*
- multiplicative models: time series = trend * seasonality * noise
- additive model: time series = trend + seasonality + noise
- stationarity - mean, variance, and autocorrelation structure do not change over time
- exogenous variable = y = dependent variable = value is determined outside the model and is imposed on the model
- endogenous variable = x = independent variable = regressor - value is determined by model

# libraries

- [pandas has some great time-series functionality](https://tomaugspurger.github.io/modern-7-timeseries)
- [skits library](https://github.com/EthanRosenthal/skits) for forecasting

# high-level modelling

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

# modelling

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