---
layout: notes
section-type: notes
title: Testing
category: stat
---
* TOC
{:toc}

---

# basics

- *data snooping* - decide which hypotheses to test after examining data
- null hypothesis $H_0$ vs alternative hypothesis $H_1$
- types
  - simple hypothesis $\theta = \theta_0$
  - composite hypothesis $\theta > \theta_0$ or $\theta < \theta_0$
  - two-sided test: $H_0: \theta = \theta_0 \: vs. \: H_1 \theta \neq \theta_0$
  - one-sided test: $H_0: \theta \leq \theta_0 \: vs. \: H_1: \theta > \theta_0$
- significance levels
  - *stat. significant*: p = 0.05
  - *highly stat. significant*: p = 0.01
- errors
  - $\alpha$ - type 1 - reject $H_0$ but $H_0$ true
  - $\beta$ - type 2 - fail to reject $H_0$ but $H_0$ false
- *p-value* = probability, calculated assuming that the null hypothesis is true, of obtaining a value of the test statistic at least as contradictory to $H_0$ as the value calculated from the available sample
- *power*: $1 - \beta$
- adjustments
  - *bonferroni procedure* - we are doing 3 tests with 5% confidence, so we actually do 5/3% for each test in order to restrict everything to 5% total
  - *Benjamini–Hochberg procedure* - controls for false discovery rate

# normal theory

- normal theory: assume $\epsilon_i$ ~ $N(0, \sigma^2)$
- distributions
  - suppose $Z_1, ..., Z_n$ ~ iid N(0, 1)
  - **chi-squared**: $\chi_d^2$ ~ $\sum_i^d U_i^2$ w/ d degrees of freedom
    - $(d-1)S^2/\sigma^2 \text{ proportional to } \chi_{d-1}^2$
  - *student's t*: $U_{d+1} / \sqrt{d^{-1} \sum_1^d U_i^2}$ w/ d degress of freedom
- **t-test**: test if mean is nonzero
  - test null $\theta_k=0$ w/ $t = \hat{\theta}_k / \hat{SE}$ where $SE = \hat{\sigma} \cdot \sqrt{\Sigma_{kk}^{-1}}$
  - t-test: reject if \|t\| is large
  - when n-p is large, t-test is called the z-test
  - under null hypothesis t follows t-distr with n-p degrees of freedom
  - here, $\hat{\theta}$ has a normal distr. with mean $\theta$ and cov matrix $\sigma^2 (X^TX)^{-1}$
    - e independent of $\hat{\theta}$ and $\|\|e\|\|^2 ~ \sigma^2 \chi^2_d$ with d = n-p
  - *observed stat. significance level* = *P-value* - area of normal curve beyond $\pm \hat{\theta_k} / \hat{SE}$
  - if 2 vars are statistically significant, said to have *independent effects* on Y
- **f-test**: test if any of non-zero means
  - null hypothesis: $\theta_i = 0,  i=p-p_0, ..., p$
  - alternative hypothesis: for at least one $ i \in \{p-p_0, ..., p\}, \: \theta_i \neq 0$
  - $F = \frac{(\|\|X\hat{\theta}\|\|^2 - \|\|X\hat{\theta}^{(s)}\|\|^2) / p_0}{\|\|e\|\|^2 / (n-p)} $ where $\hat{\theta^{(s)}}$ has last $p_0$ entries 0
  - under null hypothesis, $\|\|X\hat{\theta}\|\|^2 - \|\|X\hat{\theta}^{(s)}\|\|^2$ ~ $U$, $\|\|e\|\|^2$ ~ $V$, $F$ ~ $\frac{U/p_0}{V/(n-p)}$ where $ U \: indep \: V$, $U$ ~ $\sigma^2 \chi^2_{p_0}$, $V$ ~ $\sigma^2 \chi_{n-p}^2$
  - there is also a *partial f-test*

# statistical intervals

- interval estimates come with confidence levels
- $Z=\frac{\bar{X}-\mu}{\sigma / \sqrt{n}}$
- For p not close to 0.5, use Wilson score confidence interval (has extra terms)
- **confidence interval** - if multiple samples of trained typists were selected and an interval constructed for each sample mean, 95 percent of these intervals contain the true preferred keyboard height
  - frequentist idea

# tests on hypotheses

- Var($\bar{X}-\bar{Y})=\frac{\sigma_1^2}{m}+\frac{\sigma_2^2}{n}$
- tail refers to the side we reject (e.g. upper-tailed=$H_a:\theta>\theta_0$
- we try to make the null hypothesis a statement of equality
- upper-tailed - reject large values
- $\alpha$ is computed using the probability distribution of the test statistic when $H_0$ is true, whereas determination of b requires knowing the test statistic distribution when $H_0$ is false
- type 1 error usually more serious, pick $\alpha$ level, then constrain $\beta$
- can standardize values and test these instead

# testing LR coefficients

- confidence interval construction
  - confidence interval (CI) is range of values likely to include true value of a parameter of interest
  - confidence level (CL) - probability that the procedure used to determine CI will provide an interval that covers the value of the parameter -  if we remade it 100 times, 95 would contain the true $\theta_1$
- $\hat{\beta_0} \pm t_{n-2,\alpha /2} * s.e.(\hat{\beta_0}) $
  - for $\beta_1$
    - with known $\sigma$
      - $\frac{\hat{\beta_1}-\beta_1}{\sigma(\hat{\beta_1})} \sim N(0,1)$
      - derive CI
    - with unknown $\sigma$
      - $\frac{\hat{\beta_1}-\beta_1}{s(\hat{\beta_1})} \sim t_{n-2}$
      - derive CI

# ANOVA (analysis of variance)

- y - called dependent, response variable
- x - independent, explanatory, predictor variable
- notation: $E(Y\|x^*) = \mu_{Y\cdot x^*} = $ mean value of Y when x = $x^*$
- Y = f(x) + $\epsilon$
- linear: $Y=\beta_0+\beta_1 x+\epsilon$
- logistic: $odds = \frac{p(x)}{1-p(x)}=e^{\beta_0+\beta_1 x+\epsilon}$
- we minimize least squares: $SSE = \sum_{i=1}^n (y_i-(b_0+b_1x_i))^2$
- $b_1=\hat{\beta_1}=\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sum (x_i-\bar{x})^2} = \frac{S_{xy}}{S_{xx}}$
- $b_0=\bar{y}-\hat{\beta_1}\bar{x}$
- $S_{xy}=\sum x_iy_i-\frac{(\sum x_i)(\sum y_i)}{n}$
- $S_{xx}=\sum x_i^2 - \frac{(\sum x_i)^2}{n}$
- residuals: $y_i-\hat{y_i}$
- SSE = $\sum y_i^2 - \hat{\beta}_0 \sum y_i - \hat{\beta}_1 \sum x_iy_i$
- SST  = total sum of squares = $S_{yy} = \sum (y_i-\bar{y})^2 = \sum y_i^2 - (\sum y_i)^2/n$
- $r^2 = 1-\frac{SSE}{SST}=\frac{SSR}{SST}$ - proportion of observed variation that can be explained by regression
- $\hat{\sigma}^2 = \frac{SSE}{n-2}$
- $T=\frac{\hat{\beta}_1-\beta_1}{S / \sqrt{S_{xx}}}$ has a t distr. with n-2 df
- $s_{\hat{\beta_1}}=\frac{s}{\sqrt{S_{xx}}}$
- $s_{\hat{\beta_0}+\hat{\beta_1}x^*} = s\sqrt{\frac{1}{n}+\frac{(x^*-\bar{x})^2}{S_{xx}}}$
- sample correlation coefficient $r = \frac{S_{xy}}{\sqrt{S_xx}\sqrt{S_{yy}}}$
- this is a point estimate for population correlation coefficient = $\frac{Cov(X,Y)}{\sigma_X\sigma_Y}$
- make fisher transformation - this test statistic also tests correlation
- degrees of freedom
- one-sample T = n-1
- T procedures with paired data - n-1
- T procedures for 2 independent populations - use formula ~= smaller of n1-1 and n2-1
- variance - n-2
- use z-test if you know the standard deviation---