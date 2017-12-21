

[TOC]

# Multivariate Gaussians (Jordan  13)

### 13.1 - parameterizations

- $x \in \mathbb{R}^n$

1. *canonical parameterization*: $$p(x\|\mu, \Sigma) = \frac{1}{(2\pi )^{n/2} \|\Sigma\|^{1/2}} exp\left( -\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right)$$
2. *momemt parameterization*: $$p(x\|\eta, \Lambda) = exp\left( a + \eta^T x - \frac{1}{2} x^T \Lambda x\right)$$ ~ also called information parameterization
  - $\Lambda = \Sigma^{-1}$
  - $\eta = \Sigma^{-1} \mu$

### 13.2 - joint distributions

- split parameters into block matrices

### 13.3 - partitioned matrices

- want to *block diagonalize* the matrix
- *Schur complement* of matrix M w.r.t. H: $M/H$

### 13.4 - marginalization and conditioning

- $\mu = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}$
- $\Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12}\\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}$
- factor $p(x_1, x_2) = p(x_1|x_2)\:p(x_2) = conditional * marginal$
  - marginal
    - $\mu_2^m = \mu_2$
    - $\Sigma_2^m = \Sigma_{22}$
  - conditional
    - $\mu_{1|2}^c = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1} (x_2 - \mu_2)$
    - $\Sigma_{1|2}^c = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}$

### 13.5 - mle

- trick with the trace for taking derivs: $x^TAx = tr[x^TAx] = tr[xx^TA]$
  - $\frac{\partial}{\partial A} x^TAx = \frac{\partial}{\partial A} tr[xx^TA] = [xx^T]^T = xx^T$
- we can calculate derivs of quadratic forms by calculating derivs of traces
- useful result: $\frac{\partial}{\partial A} log|A| = A^{-T}$

# Kalman Filtering and Smoothing (Jordan 15)

### 15.1 - state space model
- underlying *state space model* = SSM is structurally identical to HMM
  - type of nodes (real-valued vectors) and prob model (linear-Gaussian) changes	- state nodes: $x_{t+1} = Ax_t + Gw_t$
    - x is Gaussian, w is noise Gaussian
  - output nodes: $y_t = Cx_t+v_t$
    - y is linear Gaussian

### 15.2 - unconditional distr.
### 15.3 - inference

### 15.4 - filtering

### 15.5 - interpretation and relationship to LMS

### 15.6 - information filter

### 15.7 - smoothing

### 15.8 - parameter estimation