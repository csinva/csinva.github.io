---
layout: notes
title: Stat basics
category: stat
subtitle: general notes on stat
---

{:toc}

# basics

- mutually exclusive: $P(AB)=0$
- independence $A \perp B$ means $P(AB) = P(A)P(B)$
  - conditional independence $$A \perp B\:\vert \:C$$ means $P(AB\vert C) = P(A\vert C) P(B\vert C)$
- conditional prob: $$P(A\vert B) = \frac{P(AB)}{P(B)} = \frac{P(B\vert A)P(A)}{\sum P(B\vert A)P(A)}$$ (Bayes' thm)
- $E[X] = \int P(x)x dx$
  - $E[h(X)] \approx h(E[X])$
- $V[X] = E[(x-\mu)^2] = E[x^2]-E[x]^2$
  - for unbiased estimate, divide by n-1
  - $V(X_1-X_2) = V(X_1) + V(X_2)$ if $X_1,X_2$ independent
  - $$V(a_1X_1+...+a_nX_n) =  \sum_{i=1}^{n}\sum_{j=1}^{n}a_ia_jcov(X_i,X_j)$$<!-- -->
  - $$V[h(X)] \approx h'(E[X])^2 V[X]$$<!-- -->
  - standard deviation $\sigma(X)=\sqrt{V[X]}$
  - standard error = error of the mean = $\sigma / \sqrt n$
- $Cov[X,Y] = E[(X-\mu_X)(Y-\mu_Y)] = E[XY]-E[X]E[Y]$
  - $Cov(aX+bY,Z) = aCov(X,Z)+bCov(Y,Z)$
- $Corr(Y,X) = \rho = \frac{Cov(Y,X)}{s_xs_y}$
  - $Corr(aX+b,cY+d) = Corr(X,Y)$ if a and c have same sign
  - $R^2 = \rho^2$
- skewness = $E[(\frac{X-\mu}{\sigma})^3]$
- law of total expectation: $$E[X] = E_Y[E(X\vert Y)]$$
- law of total variance: $$V[Y] =\cblue{\underbrace{E[V(Y\vert X)]}_{\text{unexplained variance}}} + \corange{\underbrace{V(E[Y\vert X])}_{\text{explained variance}}}$$

## error bars

- always write what you use: standard dev, standard error, or 95% confidence interval = 1.96 * standard error
- can get prediction intervals for on-line data using [conformal prediction](https://arxiv.org/abs/0706.3188)
  - *nonconformity measure* - how unusual an examples looks relative to previous examples

## inter-rater agreement

- cohen's kappa - measures how well different raters agree (just taking fraction may be too simple, because they might agree by chance)
  - from -1 to 1 (1 is perfect agreement)
  - $\kappa = 1 - \frac{1 - p_o}{1-p_e}$ where $p_o$ is the relative observed agreement among raters (identical to [accuracy](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers)), and $p_e$ is the hypothetical probability of chance agreement, using the observed data to calculate the probabilities of each observer randomly seeing each category

## inequalities

- *cauchy-schwarz*: $$\vert x \cdot y\vert  \leq \vert \vert x\vert \vert \:\vert \vert y\vert \vert $$
  - $E[XY]^2 \leq E[X^2] E[Y^2]$
- *jensen's*: $f(E[X]) \leq E[f(X)]$ for convex $f$
- *triangle*: $\vert \vert x + y \vert \vert \leq \vert \vert x \vert \vert + \vert \vert y \vert \vert$
- *markov's*: $P(X \geq a) \leq \frac{E[X]}{a}$
  - X is typically running time of the algorithm
  - if we don't have E[X], can use upper bound for E[X]
- *chebyshev's*: $P(\vert X-\mu\vert  \geq a) \leq \frac{Var[X]}{a^2}$
  - utilizes the variance to get a better bound

## moment-generating function

- $M_X(t) = E(e^{tX})$
  - derivatives yield moments: $\frac{d^r}{dX^r} M_X (0) = E(X^r)$
- sometimes $\ln[M_x(t)]$ yields $\mu$ and $V(X)$
- $Y = aX+b \implies M_y(t) = e^{bt}M_x(at)$
- $Y = a_1X_1+a_2X_2 \implies M_Y(t) = M_{X_1}(a_1t)M_{X_2}(a_2t)$ if $X_i$ independent
- *ordered statistics* - variables $Y_i$ such that $Y_i$ is the ith smalless

# distributions

- PMF: $f_X(x) = P(X=x)$
- PDF: $P(a \leq X \leq b) = \int_a^b f(x) dx$

*these distributions are from the probability and statistics cookbook by matthias vallentin*

| discrete | $$F_X(x)$$ | $$f_X(x)$$ | $$\mathbb{E}[X]$$ | $$\mathbb{V}[X]$$ | $$M_X(s)$$ |
| --- | --- | --- | --- | --- | --- |
| uniform $$\text{Unif}\{a, \dots, b\}$$ | $$\begin{cases} 0 & x < a \\ \frac{\lfloor x \rfloor - a + 1}{b - a} & a \le x \le b \\ 1 & x > b \end{cases}$$ | $$\frac{I(a \le x \le b)}{b - a + 1}$$ | $$\frac{a+b}{2}$$ | $$\frac{(b-a+1)^2 - 1}{12}$$ | $$\frac{e^{as} - e^{-(b+1)s}}{s(b-a)}$$ |
| bernoulli $$\text{Bern}(p)$$ | $$(1-p)^{1-x}$$ | $$p^x (1-p)^{1-x}$$ | $$p$$ | $$p(1-p)$$ | $$1 - p + pe^s$$ |
| binomial $$\text{Bin}(n, p)$$ | $$I_{1-p}(n-x, x+1)$$ | $$\binom{n}{x} p^x (1-p)^{n-x}$$ | $$np$$ | $$np(1-p)$$ | $$(1 - p + pe^s)^n$$ |
| multinomial $$\text{Mult}(n, p)$$ | | $$\frac{n!}{x_1! \dots x_k!} p_1^{x_1} \cdots p_k^{x_k} \quad \sum_{i=1}^k x_i = n$$ | $$np_i$$ | $$np_i(1-p_i)$$ | $$\left(\sum_{i=0}^k p_i e^{s_i}\right)^n$$ |
| hypergeometric $$\text{Hyp}(N, m, n)$$ | $$\approx \Phi\left(\frac{x - np}{\sqrt{np(1-p)}}\right)$$ | $$\frac{\binom{m}{x}\binom{N-m}{n-x}}{\binom{N}{n}}$$ | $$\frac{nm}{N}$$ | $$\frac{nm(N-n)(N-m)}{N^2(N-1)}$$ | |
| neg. binomial $$\text{NBin}(r, p)$$ | $$I_p(r, x+1)$$ | $$\binom{x+r-1}{r-1} p^r (1-p)^x$$ | $$r\frac{1-p}{p}$$ | $$r\frac{1-p}{p^2}$$ | $$\left(\frac{p}{1-(1-p)e^s}\right)^r$$ |
| geometric $$\text{Geo}(p)$$ | $$1 - (1-p)^x \quad x \in \mathbb{N}^+$$ | $$p(1-p)^{x-1} \quad x \in \mathbb{N}^+$$ | $$\frac{1}{p}$$ | $$\frac{1-p}{p^2}$$ | $$\frac{pe^s}{1-(1-p)e^s}$$ |
| poisson $$\text{Po}(\lambda)$$ | $$e^{-\lambda} \sum_{i=0}^x \frac{\lambda^i}{i!}$$ | $$\frac{\lambda^x e^{-\lambda}}{x!}$$ | $$\lambda$$ | $$\lambda$$ | $$e^{\lambda(e^s - 1)}$$ |

| continuous | $$F_X(x)$$ | $$f_X(x)$$ | $$\mathbb{E}[X]$$ | $$\mathbb{V}[X]$$ |
| --- | --- | --- | --- | --- |
| uniform $$\text{Unif}(a, b)$$ | $$\begin{cases} 0 & x < a \\ \frac{x-a}{b-a} & a < x < b \\ 1 & x > b \end{cases}$$ | $$\frac{I(a < x < b)}{b - a}$$ | $$\frac{a+b}{2}$$ | $$\frac{(b-a)^2}{12}$$ |
| normal $$\mathcal{N}(\mu, \sigma^2)$$ | $$\Phi(x) = \int_{-\infty}^x \phi(t)\, dt$$ | $$\phi(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\}$$ | $$\mu$$ | $$\sigma^2$$ |
| log-normal $$\ln\mathcal{N}(\mu, \sigma^2)$$ | $$\frac{1}{2} + \frac{1}{2} \text{erf}\left[\frac{\ln x - \mu}{\sqrt{2\sigma^2}}\right]$$ | $$\frac{1}{x\sqrt{2\pi\sigma^2}} \exp\left\{-\frac{(\ln x - \mu)^2}{2\sigma^2}\right\}$$ | $$e^{\mu + \sigma^2/2}$$ | $$(e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$$ |
| multi norm $$\text{MVN}(\mu, \Sigma)$$ | | $$(2\pi)^{-k/2} \vert\Sigma\vert^{-1/2} e^{-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)}$$ | $$\mu$$ | $$\Sigma$$ |
| student's t $$\text{Student}(\nu)$$ | $$I_x\left(\frac{\nu}{2}, \frac{\nu}{2}\right)$$ | $$\frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-(\nu+1)/2}$$ | $$0$$ | $$\begin{cases} \frac{\nu}{\nu-2} & \nu > 2 \\ \infty & 1 < \nu \le 2 \end{cases}$$ |
| chi-square $$\chi^2_k$$ | $$\frac{1}{\Gamma(k/2)} \gamma\left(\frac{k}{2}, \frac{x}{2}\right)$$ | $$\frac{1}{2^{k/2}\Gamma(k/2)} x^{k/2-1} e^{-x/2}$$ | $$k$$ | $$2k$$ |
| F $$\text{F}(d_1, d_2)$$ | $$I_{\frac{d_1 x}{d_1 x + d_2}}\left(\frac{d_1}{2}, \frac{d_1}{2}\right)$$ | $$\frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1 + d_2}}}}{x \text{B}\left(\frac{d_1}{2}, \frac{d_1}{2}\right)}$$ | $$\frac{d_2}{d_2 - 2}$$ | $$\frac{2 d_2^2 (d_1 + d_2 - 2)}{d_1 (d_2 - 2)^2 (d_2 - 4)}$$ |
| exponential $$\text{Exp}(\beta)$$ | $$1 - e^{-x/\beta}$$ | $$\frac{1}{\beta} e^{-x/\beta}$$ | $$\beta$$ | $$\beta^2$$ |
| gamma $$\text{Gamma}(\alpha, \beta)$$ | $$\frac{\gamma(\alpha, x/\beta)}{\Gamma(\alpha)}$$ | $$\frac{1}{\Gamma(\alpha)\beta^\alpha} x^{\alpha-1} e^{-x/\beta}$$ | $$\alpha\beta$$ | $$\alpha\beta^2$$ |
| inv. gamma $$\text{InvGamma}(\alpha, \beta)$$ | $$\frac{\Gamma\left(\alpha, \frac{\beta}{x}\right)}{\Gamma(\alpha)}$$ | $$\frac{\beta^\alpha}{\Gamma(\alpha)} x^{-\alpha-1} e^{-\beta/x}$$ | $$\frac{\beta}{\alpha-1} \quad \alpha > 1$$ | $$\frac{\beta^2}{(\alpha-1)^2(\alpha-2)} \quad \alpha > 2$$ |
| dirichlet $$\text{Dir}(\alpha)$$ | | $$\frac{\Gamma\left(\sum_{i=1}^k \alpha_i\right)}{\prod_{i=1}^k \Gamma(\alpha_i)} \prod_{i=1}^k x_i^{\alpha_i - 1}$$ | $$\frac{\alpha_i}{\sum_{i=1}^k \alpha_i}$$ | $$\frac{\mathbb{E}[X_i](1 - \mathbb{E}[X_i])}{\sum_{i=1}^k \alpha_i + 1}$$ |
| beta $$\text{Beta}(\alpha, \beta)$$ | $$I_x(\alpha, \beta)$$ | $$\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta-1}$$ | $$\frac{\alpha}{\alpha+\beta}$$ | $$\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$ |
| weibull $$\text{Weibull}(\lambda, k)$$ | $$1 - e^{-(x/\lambda)^k}$$ | $$\frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}$$ | $$\lambda\Gamma\left(1 + \frac{1}{k}\right)$$ | $$\lambda^2\Gamma\left(1 + \frac{2}{k}\right) - \mu^2$$ |
| pareto $$\text{Pareto}(x_m, \alpha)$$ | $$1 - \left(\frac{x_m}{x}\right)^\alpha \quad x \ge x_m$$ | $$\alpha \frac{x_m^\alpha}{x^{\alpha+1}} \quad x \ge x_m$$ | $$\frac{\alpha x_m}{\alpha - 1} \quad \alpha > 1$$ | $$\frac{x_m^\alpha}{(\alpha-1)^2(\alpha-2)} \quad \alpha > 2$$ |

- multivariate gaussian
  - 2 parameterizations ($x \in \mathbb{R}^n$)
    1. *canonical parameterization*: $$p(x\vert\mu, \Sigma) = \frac{1}{(2\pi )^{n/2} \vert\Sigma\vert^{1/2}} \exp\left[ -\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right]$$
    2. *moment parameterization*: $$p(x\vert\eta, \Omega) = \text{exp}\left( a + \eta^T x - \frac{1}{2} x^T \Omega x\right)$$ ~ also called information parameterization
       - $\Omega = \Sigma^{-1}$
       - $\eta = \Sigma^{-1} \mu$
  - joint distr - split parameters into block matrices
  - want to *block diagonalize* the matrix
    - *Schur complement* of matrix M w.r.t. H: $M/H$
    - $$\mu = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}$$<!-- -->
    - $$\Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12}\\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}$$<!-- -->
    - $$p(x_1, x_2) = \underbrace{p(x_1\vert x_2)}_{\text{conditional}}\cdot\underbrace{p(x_2)}_{\text{marginal}}$$<!-- -->
      - marginal
        - $\mu_2^m = \mu_2$
        - $\Sigma_2^m = \Sigma_{22}$
      - conditional
        - $$\mu_{1\vert 2}^c = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1} (x_2 - \mu_2)$$<!-- -->
        - $$\Sigma_{1\vert 2}^c = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}$$<!-- -->

# law of large numbers

## law of large numbers

- equivalent statements
  - $E(\bar{X}-\mu)^2 \to 0$ as $n \to \infty,$
  - $P(\vert\bar{X}-\mu\vert \geq \epsilon) \to 0$ as $n \to \infty$
  - $$T_o = X_1+...+X_n, E(T_o) = n\mu , V(T_o) = n\mu ^2$$<!-- -->
- implications
  - $E(\bar{X}) = \mu$
  - $V(\bar{X}) = \frac{\sigma_x^2}{n}$

## central limit thm

- 2 characterizations
  - random sample means have a normal distr. if n is large
  - $\lim_{n\to\infty}P(\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\leq z)=P(Z\leq z) = \Phi(z)$
- implications
  - $$X_1*..*X_n$$ has approximately lognormal distribution if all $P(X_i>0)$

# bias and point estimation

- *point estimator* $\hat{\theta}$ - statistic that predicts a parameter
  - *point estimate* - single number prediction
- **bias**: $E(\hat{\theta}) - \theta$
  - more complex models (more nonzero parameters) have lower bias, higher variance
    - if high bias, train and test error will be very close (model isn't complex enough)
  - after unbiased we want MVUE (minimum variance unbiased estimator)
  - need *inductive inference property*: must make prior assumptions in order to classify unseen instances
    - define *inductive bias* of a learner as the set of additional assumptions B sufficient to justify its inductive inferences as deductive inferences
  - bias types
    - *preference bias* = *search bias* - models can search entire space (e.g. NN, decision tree)
    - *restriction bias* = *language bias* - models that can't express entire space (e.g. linear)
- **consistent**: $$\hat{\theta_n} \to some \: value$$
  - basically it converges to a number (can still be biased)
- **bias/variance trade-off**
  - MSE - mean squared error - $E[(\hat{\theta}-\theta)^2]$ = $\cblue{V(\hat{\theta})}+\corange{[E(\hat{\theta})-\theta]^2}$
    - decomposition: $$MSE = \corange{Bias^2} + \cblue{Variance}$$

      which can be shown as follows

      $$
      \begin{aligned}
      MSE(\hat{\theta}) &= E\left[\hat{\theta} - \theta\right]^2 \\
      &= E\left[\hat{\theta} - E(\hat{\theta}) + E(\hat{\theta}) - \theta\right]^2 \quad \text{(add and subtract } E(\hat{\theta})\text{)} \\
      &= \cblue{E\left[\hat{\theta} - E(\hat{\theta})\right]^2} + \cgreen{2E\left[(\hat{\theta} - E(\hat{\theta}))(E(\hat{\theta}) - \theta)\right]} \\
      &\quad + \corange{E\left[E(\hat{\theta}) - \theta\right]^2} \\
      &= \cblue{Var(\hat{\theta})} + \cgreen{2(E(\hat{\theta}) - \theta)E\left[\hat{\theta} - E(\hat{\theta})\right]} \\
      &\quad + \corange{\left(E(\hat{\theta} - \theta)\right)^2} \quad \text{(} E(\hat{\theta}) \text{ and } \theta \text{ are constant)} \\
      &= \cblue{Var(\hat{\theta})} + \corange{Bias(\hat{\theta})^2}
      \end{aligned}
      $$

      where the last equality follows from <span class="cgreen">the cross term vanishing</span>: $$\cgreen{E\left[\hat{\theta} - E(\hat{\theta})\right]} = E(\hat{\theta}) - E(\hat{\theta}) = 0$$

  - defs
    - <span class="corange">bias</span> = *approximation err*
    - <span class="cblue">variance</span> = *estimation err*
- **confidence intervals** - take sample data + produce a range of values that likely contains population parameter you are interested in
- [confidence interval of the prediction](https://statisticsbyjim.com/glossary/confidence-interval-prediction/) is a range that likely contains the mean value of the dependent variable given specific values of the independent variables - usually wider because it is one point, not a mean
- MLE example
  - MLE - maximize likelihood $$L(\theta) = p(X_1,...,X_n;\theta_1,...\theta_m)$$ (the agreement with a chosen distribution)
  - $\hat{\theta} = $argmax $  L(\theta)$
      - $$L(\theta)=P(X_1...X_n\vert\theta)=\prod_{i=1}^n P(X_i\vert\theta)$$<!-- -->
      - $$\log \: L(\theta)= \ell(\theta) = \sum \log P(X_i\vert\theta)$$<!-- -->
      - to maximize, set $\frac{\partial \ell (\theta)}{\partial \theta} = 0$
  - **fisher information** $I(\theta)=V[\frac{\partial^2}{\partial\theta^2} \overbrace{\ln(f[x;\theta])}^{\text{Fisher score function}} ]$ (for n samples, multiply by n)
      - higher info $\implies$ lower estimation error
      - $\overbrace{\ln(f[x;\theta])}^{\text{Fisher score function}}$

# overview - Jordan ch 5

from Michael Jordan's "An Introduction to Probabilistic Graphical Models"

- prob theory: given model $\theta$, infer data $X$
- statistics: given data $X$, infer model $\theta$
- 2 statistical schools of thought: *Bayesian* and *frequentist*
  1. Bayesian: $\cgreen{\overbrace{p(\theta \vert x)}^{\text{posterior}}} = \frac{\cblue{\overbrace{p(x\vert\theta)}^{\text{likelihood}}} \corange{\overbrace{p(\theta)}^{\text{prior}}}}{p(x)}$
     - assumes $\theta$ is a RV, find its distr.
     - <span class="corange">prior probability</span> $\corange{p(\theta)}$= *statistician's uncertainty*
       - *<span class="cgreen">posterior</span>* $$\cgreen{p(\theta\vert x)}$$ is what you don't observe
     - $$\hat{\theta}_{Bayes} = \int \theta \: p(\theta \vert x) d\theta$$ ~ mean of the posterior
     - $$\hat{\theta}_{MAP} = \underset{\theta}{\operatorname*{argmax}} \: \cgreen{p(\theta\vert x)} = \underset{\theta}{\operatorname*{argmax}} \: \cblue{p(x\vert \theta)} \corange{p(\theta)}  \\\ = \underset{\theta}{\operatorname*{argmax}} \: [ \cblue{\log \: p(x\vert\theta)} + \corange{\log \: p(\theta)} ]$$<!-- -->
       - like *penalized likelihood*: <span class="cblue">log-likelihood</span> + <span class="corange">log-prior</span> penalty
     - bayesians prefer whole distr. rather than parameter estimates
  2. frequentist - use estimators (ex. MLE)
     - no prior - only use priors when they correspond to objective frequencies of observing values
     - neyman / pearson
     - $$\hat{\theta}_{MLE} = \operatorname*{argmax}_\theta \: p(x\vert\theta)$$<!-- -->
       - really likelihood is whatever we model (ex. for discriminative models would be $$p(y\vert x, \theta)$$)

## 3 problems

1. *density estimation* - given samples of X, estimate P(X)
   - ex. univariate Gaussian density estimation
     - frequentist
       - derive MLE for mean and variance
     - bayesian
       - assume distr. for $\mu$  
         - ex. $p(\mu) \sim N(\mu_0, \tau^2)$
       - derive MAP for mean and variance (assuming some prior)
     - can use plate to show repeated element
   - ex. discrete, multinomial prob. distr.
     - derive MLE
       - $$P(x\vert \theta) \sim $$multionomial distr.
     - derive MAP
       - want to be able to plug in posterior as prior recursively
       - this requires a *Dirichlet prior* to multiply the multinomial
         - Dirichlet: $p(\theta) = C(\alpha) \theta_1^{\alpha_1 - 1}\cdot \cdot \cdot \theta_M^{\alpha_M-1}$
   - ex. *mixture models* - $p(x\vert\theta)=\sum_k \alpha_k f_k (x\vert\theta_k)$
     - here $f_k$ represent densities (*mixture components*)
     - $\alpha_k$ are weights (*mixing proportions*)
     - can do inference on this - given x, figure out which cluster it fits into better
     - learning requires EM
     - can be used nonparametrically - *mixture seive*
       - however, means are allowed to vary
     - solving with random projection: project to low dim and keep track of means etc.
   - ex. *nonparametric density estimation*
     - ex. *kernel density estimator* - stacking up mass
     - each point contributes a kernel function $k(x,x_n, \lambda)$
       - $x_n$ is location, $\lambda$ is smoothing
     - $\hat{p}(x) = \frac{1}{N}\sum_n k(x,x_n,\lambda)$
     - nonparametric models sometimes called *infinite-dimensional*
2. *regression* - want $p(y \vert x)$
   - *conditional mixture model* - variable z can be used to pick out regions of input space where different regression functions are used
     - $p(y_n\vert x_n,\theta) = \sum_k p(y_n\vert z_n^k = 1, x_n, \theta) \cdot  p(z_n^k=1\vert x_n,\theta)$
   - *nonparametric regression*
     - ex. *kernel regression* $\hat{f}(x) = \frac{\sum_{i=1}^N k(x, x_i) \cdot y_i}{\sum_{m=1}^N k(x, x_j)}$
3. *classification*
   - ex. Gaussian class-conditional densities
     - posterior probability is *logistic function*
   - *clustering* - use mixture models

## model selection / averaging

- bayesian
  - for model m, want to maximize $p(m\vert x) = \frac{p(x\vert m) p(m)}{p(x)}$
    - usually, just take $m$ that maximizes $p(m\vert x)$
    - *model averaging*: $$p(x_{new}\vert x) = \int dm \int d\theta \: p(x_{new}\vert \theta, m) p(\theta\vert x, m) p(m\vert x)$$
    - otherwise integrate over $\theta, m$ - *model averaging*
- frequentist
  - can't use MLE - will always prefer more complex models
  - use some criteria such as KL-divergence, AIC, cross-validation