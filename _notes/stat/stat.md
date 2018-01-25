---
layout: notes
section-type: notes
title: Statistics
category: stat
---
* TOC
{:toc}
- *material based on probability and statistics cookbook by matthias vallentin*

# probability

- mutually exclusive: $P(AB)=0$
- independent: $P(AB) = P(A)P(B)$
  - A and B conditional independence given C: $$P(AB\vert C) = P(A\vert C) P(B\vert C)$$
- conditional (Bayes' thm): $P(A|B) = \frac{P(AB)}{P(B)} = \frac{P(B|A)P(A)}{\sum P(B|A)P(A)}$

# distributions

- PMF: $f_X(x) = P(X=x)$
- PDF: $P(a \leq X \leq b) = \int_a^b f(x) dx$

![distrs](assets/stat/distrs.png)



## multivariate gaussians - j 13

- 2 parameterizations
  - $x \in \mathbb{R}^n$

1. *canonical parameterization*: $$p(x\|\mu, \Sigma) = \frac{1}{(2\pi )^{n/2} \|\Sigma\|^{1/2}} exp\left[ -\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right]$$
2. *moment parameterization*: $$p(x\|\eta, \Lambda) = \text{exp}\left( a + \eta^T x - \frac{1}{2} x^T \Lambda x\right)$$ ~ also called information parameterization
   - $\Lambda = \Sigma^{-1}$
   - $\eta = \Sigma^{-1} \mu$

- joint distr - split parameters into block matrices


- want to *block diagonalize* the matrix
  - *Schur complement* of matrix M w.r.t. H: $M/H$
  - $\mu = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}$
  - $\Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12}\\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}$
  - $p(x_1, x_2) = p(x_1|x_2)\:p(x_2) = conditional * marginal$
    - marginal
      - $\mu_2^m = \mu_2$
      - $\Sigma_2^m = \Sigma_{22}$
    - conditional
      - $\mu_{1|2}^c = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1} (x_2 - \mu_2)$
      - $\Sigma_{1|2}^c = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}$
- mle
  - trick with the trace for taking derivs: $x^TAx = tr[x^TAx] = tr[xx^TA]$
    - $\frac{\partial}{\partial A} x^TAx = \frac{\partial}{\partial A} tr[xx^TA] = [xx^T]^T = xx^T$
  - we can calculate derivs of quadratic forms by calculating derivs of traces
  - useful result: $\frac{\partial}{\partial A} log|A| = A^{-T}$

# expectation and variance

- $E[X] = \int P(x)x dx$
- $V[X] = E[(x-\mu)^2] = E[x^2]-E[x]^2$
  - for unbiased estimate, divide by n-1
- $Cov[X,Y] = E[(X-\mu_X)(Y-\mu_Y)] = E[XY]-E[X]E[Y]$
  - $Cor(Y,X) = \rho = \frac{Cov(Y,X)}{s_xs_y}$
  - $R^2 = \rho^2$
- linearity
  - $Cov(aX+bY,Z) = aCov(X,Z)+bCov(Y,Z)$
  - $V(a_1X_...+a_nX_n) =  \sum_{i=1}^{n}\sum_{j=1}^{n}a_ia_jcov(X_i,X_j)$
  - if $X_1,X_2$ independent, $V(X_1-X_2) = V(X_1) + V(X_2)$
- $f(X), X=v(Y), g(Y) = f(v(Y))$ \|$\frac{d}{dy}g^{-1}(y)$\|
- $g(y_1,y_2) = f(v_1,v_2)\|det(M)\|$ where M in row-major is $\frac{\partial v1}{y1}, \frac{\partial v1}{y2} ...$
- $Corr(aX+b,cY+d) = Corr(X,Y)$ if a and c have same sign
- $E[h(X)] \approx h(E[X])$
- $V[h(X)] \approx h'(E[X])^2 V[X]$
- skewness = $E[(\frac{X-\mu}{\sigma})^3]$

# inequalities

- *Cauchy-Schwarz*: $E[XY]^2 \leq E[X^2] E[Y^2]$
- *Markov's*: $P(X \geq a) \leq \frac{E[X]}{a}$
  - X is typically running time of the algorithm
  - if we don't have E[X], can use upper bound for E[X]
- *Chebyshev's*: $P(\vert X-\mu\vert  \geq a) \leq \frac{Var[X]}{a^2}$
  - utilizes the variance to get a better bound
- *Jensen's*: $f(E[X]) \leq E[f(X)]$ for convex f

# moment-generating function

- $M_X(t) = E(e^{tX})$
- $E(X^r) = M_X ^ {(r )} (0)$
- sometimes you can use $ln[M_x(t)]$ to find $\mu$ and $V(X)$
- Y = aX+b $-> M_y(t) = e^{bt}M_x(at)$
- Y = $a_1X_1+a_2X_2 \to M_Y(t) = M_{X_1}(a_1t)M_{X_2}(a_2t)$ if $X_i$ independent
- probability plot  - straight line is better - plot ([100(i-.5)/n]the percentile, ith ordered observation)
- ordered statistics - variables $Y_i$ such that $Y_i$ is the ith smallest
- If X has pdf f(x) and cdf F(x), $G_n(y) = (F(y))^n$, $g_n(y) = n(F(y))^{n-1}f(y)$
  - If joint, $g(y_1,...y_n) = n!f(y_1)...f(y_n)$

  $g(y_i) = \frac{n!}{(i-1)!(n-i)!}(F(y_i))^{i-1}(1-F(y_i))^{n-1}f(y_i)​$

# statistics and sampling distributions

## law of large numbers

- $ E(\bar{X}-\mu)^2 \to 0$ as $n \to \infty,$
- $ P(\|\bar{X}-\mu\| \geq \epsilon) \to 0$ as $n \to \infty$
- $T_o = X_1+...+X_n, E(T_o) = n\mu , V(T_o) = n\mu ^2$
- $E(\bar{X}) = \mu$
- $V(\bar{X}) = \frac{\sigma_x^2}{n}$
- chi-squared - finding the distribution for sums of squares of normal variables. 
- if $Z_1,..., Z_n$ are i.i.d. standard normal,then $Z_1^2+...+Z_n^2 = \chi_n^2$
- $(n-1)S^2/\sigma^2 \text{ proportional to } \chi_{n-1}^2$
- t - to use the sample standard deviation to measure precision for the mean X, we combine the square root of a chi-squared variable with a normal variable
- f - compare two independent sample variances in terms of the ratio of two independent chi-squared variables.

## central limit thm

- CLT - random samples have a normal distr. if n is large
- CLT  has approximately lognormal distribution if all
- CLT: $lim_{n->\infty}P(\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\leq z)=P(Z\leq z) = \Phi(z)$
- CLT $\to Y = X_1*..*X_n$ has approximately lognormal distribution if all $P(X_i>0)$

# bias and point Estimation

- *point estimator* $\hat{\theta}$ - statistic that predicts a parameter
  - *point estimate* - single number prediction
- *bias*: $E(\hat{\theta}) - \theta​$
  - after unbiased we want MVUE (minimum variance unbiased estimator)
  - need *inductive inference property*: must make prior assumptions in order to classify unseen instances
  - define *inductive bias* of a learner as the set of additional assumptions B sufficient to justify its inductive inferences as deductive inferences
  - *preference bias* = *search bias* - models can search entire space (e.g. NN, decision tree)
  - *restriction bias* = *language bias* - models that can't express entire space (e.g. linear)
  - more complex models (more nonzero parameters) have lower bias, higher variance
    - if high bias, train and test error will be very close (model isn't complex enough)
- *consistency*: $\hat{\theta_n} \to \theta$
- standard error: $\sigma_{\hat{\theta}} = s_{\hat{\sigma}} = \sqrt{Var(\hat{\theta)}}$ 
- *bias/variance trade-off*
  - MSE - mean squared error  - $E[(\hat{\theta}-\theta)^2]$ = $V(\hat{\theta})+[E(\hat{\theta})-\theta]^2$
    - ![mse](assets/stat/mse.png)
  - defs
    - bias sometimes called approximation err
    - variance called estimation err

## MLE

- MLE - maximize $f(x_1,...,x_n;\theta_1,...\theta_m)​$ - agreement with chosen distribution - often take ln(f) and then take derivative $\approx​$ MVUE, but can be biased
- $\hat{\theta} = $argmax $  L(\theta)$
    - Likelihood $L(\theta)=P(X_1...X_n\|\theta)=\prod_{i=1}^n P(X_i\|\theta)$
    - $logL(\theta)=\sum log P(X_i\|\theta)$
    - to maximize, set $\frac{\partial log \: L(\theta)}{\partial \theta} = 0$
- Use $\hat{\theta} = $argmax $  P(\text{Train} \| {Model}(\theta))$
- Fisher Information $I(\theta)=V[\frac{\partial}{\partial\theta}ln(f(x;\theta))]$ (for n samples, multiply by n)

# overview - J. 5

- prob theory: given model $\theta$, infer data $X$
- statistics: given data $X$, infer model $\theta$
- 2 statistical schools of thought: *Bayesian* and *frequentist*
  1. Bayesian: $\overbrace{p(\theta \| x)}^{\text{posterior}} = \frac{\overbrace{p(x\|\theta)}^{\text{likelihood}} \overbrace{p(\theta)}^{\text{prior}}}{p(x)}$
     - assumes $\theta$ is a RV, find its distr.
     - prior probability $p(\theta)$= *statistician's uncertainty*
       - *posterior* $p(\theta|x)$ is what you don't observe
     - $\hat{\theta}_{Bayes} = \int \theta \: p(\theta\|x) d\theta$ ~ mean of the posterior
     - $\hat{\theta}_{MAP} = \underset{\theta}{argmax} \: p(\theta\|x) = \underset{\theta}{argmax} \: p(x\|\theta) p(\theta)  \\\ = \underset{\theta}{argmax} \: [ log \: p(x\|\theta) + log \: p(\theta) ]$
       - like *penalized likelihood*
     - however bayesian's don't like using these parameter estimates too much - prefer to consider the whole distributions
  2. frequentist - use estimators (ex. MLE)
     - ignores prior - only use priors when they correspond to objective frequencies of observing values
     - $\hat{\theta}_{MLE} = argmax_\theta \: p(x\|\theta)$

## 3 problems

1. *density estimation* - given samples of X, estimate P(X)
   - ex. univariate Gaussian density estimation
     - frequentist
       - derive MLE for mean and variance
     - bayesian
       - assume distr. for $\mu$ ~ ex. $p(\mu) \sim N(\mu_0, \tau^2)$
       - derive MAP for mean and variance (assuming some prior)
     - can use plate to show repeated element
   - ex. discrete, multinomial prob. distr.
     - derive MLE
       - $P(x|\theta) \sim $multionomial distr.
     - derive MAP
       - want to be able to plug in posterior as prior recursively
       - this requires a *Dirichlet prior* to multiply the multinomial
         - Dirichlet: $p(\theta) = C(\alpha) \theta_1^{\alpha_1 - 1}\cdot \cdot \cdot \theta_M^{\alpha_M-1}$
         - ex. *mixture models* - $p(x\|\theta)=\sum_k \alpha_k f_k (x\|\theta_k)$
   - ex. mixture models
     - here $f_k$ represent densities (*mixture components*)
     - $\alpha_k$ are weights (*mixing proportions*)
     - can do inference on this - given x, figure out which cluster it fits into better
     - learning requires EM
     - can be used nonparametrically - *mixture seive*
       - however, means are allowed to vary
     - solving method ex. - random projection - project ot llow dim and keep track of means etc.
   - ex. *nonparametric density estimation*
     - ex. *kernel density estimator* - stacking up mass
     - each point contributes a kernel function $k(x,x_n, \lambda)$
       - $x_n$ is location, $\lambda$ is smoothing
     - $\hat{p}(x) = \frac{1}{N}\sum_n k(x,x_n,\lambda)$
     - nonparametric models sometimes called *infinite-dimensional*
2. *regression* - want p(y\|x)
   - *conditional mixture model* - variable z can be used to pick out regions of input space where different regression functions are used
     - $p(y_n\|x_n,\theta) = \sum_k p(z_n^k=1\|x_n,\theta) \cdot p(y_n\|z_n^k = 1, x_n, \theta)$
   - *nonparametric regression* 
     - ex. *kernel regression* $\hat{f}(x) = \frac{\sum_{i=1}^N k(x, x_i) \cdot y_i}{\sum_{m=1}^N k(x, x_j)}$
3. *classification*
   - ex. Gaussian class-conditional densities
     - posterior probability is *logistic function*
   - *clustering* - use mixture models

## model selection / averaging

- bayesian
  - for model m, want to maximize $p(m\|x) = \frac{p(x\|m) p(m)}{p(x)}$
    - usually, just take $m$ that maximizes $p(m\|x)$
    - *model averaging*: $p(x_{new}|x) = \int \int p(x_{new}|\theta, m) p(\theta|x, m) p(m|x) d\theta dm$
    - otherwise integrate over $\theta, m$ - *model averaging*
- frequentist
  - can't use MLE - will always prefer more complex models
  - use some criteria such as KL-divergence, AIC, cross-validation