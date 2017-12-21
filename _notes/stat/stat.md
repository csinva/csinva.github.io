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

- Mutually Exclusive: $P(AB)=0$
- Independent: $P(AB) = P(A)P(B)$
  - A and B conditional independence given C: $$P(AB\vert C) = P(A\vert C) P(B\vert C)$$
- Conditional (Bayes' thm): $P(A\|B) = \frac{P(AB)}{P(B)} = \frac{P(B|A)P(A)}{\sum P(B|A)P(A)}$

# distributions

- PMF: $f_X(x) = P(X=x)$
- PDF: $P(a \leq X \leq b) = \int_a^b f(x) dx$

![distrs](assets/stat/distrs.png)

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
- *Markov's*
  - $P(X \geq a) \leq \frac{E[X]}{a}$
  - X is typically running time of the algorithm
  - if we don't have E[X], can use upper bound for E[X]
- *Chebyshev's*
  - $P(\vert X-\mu\vert  \geq a) \leq \frac{Var[X]}{a^2}$
  - utilizes the variance to get a better bound
- *Jensen's*: $f(E[X]) \leq E[f(X)]$ for convex f

# moment-generating function

- $M_X(t) = E(e^{tX})$
- $E(X^r) = M_X ^ {(r )} (0)$
- sometimes you can use $ln(M_x(t))$ to find $\mu$ and $V(X)$
- Y = aX+b $-> M_y(t) = e^{bt}M_x(at)$
- Y = $a_1X_1+a_2X_2 \to M_Y(t) = M_{X_1}(a_1t)M_{X_2}(a_2t)$ if $X_i$ independent
- probability plot  - straight line is better - plot ([100(i-.5)/n]the percentile, ith ordered observation)
- ordered statistics - variables $Y_i$ such that $Y_i$ is the ith smallest
- If X has pdf f(x) and cdf F(x), $G_n(y) = (F(y))^n$, $g_n(y) = n(F(y))^{n-1}f(y)$
  - If joint, $g(y_1,...y_n) = n!f(y_1)...f(y_n)$

  $g(y_i) = \frac{n!}{(i-1)!(n-i)!}(F(y_i))^{i-1}(1-F(y_i))^{n-1}f(y_i)​$

# Statistics and Sampling Distributions

## Law of Large Numbers

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

# Bias and point Estimation

- *point estimator* $\hat{\theta}$ - statistic that predicts a parameter
  - *point estimate* - single number prediction
- *bias*: $E(\hat{\theta}) - \theta$
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
