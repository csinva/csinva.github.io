[toc]

# Multivariate Gaussians (Jordan  13)
### 13.1 - parameterizations
1. $$p(x|\mu, \Sigma) = \frac{1}{(2\pi )^{n/2} |\Sigma|^{1/2}} exp\left( -\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right)$$
2. $$p(x|\eta, \Lambda) = exp\left( a + \eta^T x - \frac{1}{2} x^T \Lambda x\right)$$
	- $\Lambda = \Sigma^{-1}$
	- $\eta = \Sigma^{-1} \mu$

### 13.2 - joint distributions
### 13.3 - partitioned matrices
### 13.4 - marginalization and conditioning
### 13.5 - mle

# Kalman Filtering and Smoothing (Jordan 15)
### 15.1 - state space model
### 15.2 - unconditional distr.
### 15.3 - inference
### 15.4 - filtering
### 15.5 - interpretation and relationship to LMS
### 15.6 - information filter
### 15.7 - smoothing
### 15.8 - parameter estimation