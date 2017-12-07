---
layout: notes
section-type: notes
title: Testing
category: stat
---
* TOC
{:toc}
# basics

- *data snooping* - decide which hypotheses to test after examining data

# normal theory

- normal theory: assume $\epsilon_i$ ~ $N(0, \sigma^2)$
- distributions
  - suppose $U_1, ...$ are iid N(0, 1)
  - *chi-squared distr.*: $\chi_d^2$ ~ $\sum_i^d U_i^2$ w/ d degrees of freedom
  - *student's t-distr.*: $U_{d+1} / \sqrt{d^{-1} \sum_1^d U_i^2}$ w/ d degress of freedom
- t-test
  - test null $\theta_k=0$ w/ $t = \hat{\theta}_k / \hat{SE}$ where $SE = \hat{\sigma} \cdot \sqrt{\Sigma_{kk}^{-1}}$
  - t-test: reject if \|t\| is large
  - when n-p is large, t-test is called the z-test
  - under null hypothesis t follows t-distr with n-p degrees of freedom
  - here, $\hat{\theta}$ has a normal distr. with mean $\theta$ and cov matrix $\sigma^2 (X^TX)^{-1}$
    - e independent of $\hat{\theta}$ and $\|\|e\|\|^2 ~ \sigma^2 \chi^2_d$ with d = n-p
  - *observed stat. significance level* = *P-value* - area of normal curve beyond $\pm \hat{\theta_k} / \hat{SE}$
  - if 2 vars are statistically significant, said to have *independent effects* on Y
- the F-test
  - null hypothesis: $\theta_i = 0,  i=p-p_0, ..., p$
  - alternative hypothesis: for at least one $ i \in \{p-p_0, ..., p\}, \: \theta_i \neq 0$
  - $F = \frac{(\|\|X\hat{\theta}\|\|^2 - \|\|X\hat{\theta}^{(s)}\|\|^2) / p_0}{\|\|e\|\|^2 / (n-p)} $ where $\hat{\theta^{(s)}}$ has last $p_0$ entries 0
  - under null hypothesis, $\|\|X\hat{\theta}\|\|^2 - \|\|X\hat{\theta}^{(s)}\|\|^2$ ~ $U$, $\|\|e\|\|^2$ ~ $V$, $F$ ~ $\frac{U/p_0}{V/(n-p)}$ where $ U \: indep \: V$, $U$ ~ $\sigma^2 \chi^2_{p_0}$, $V$ ~ $\sigma^2 \chi_{n-p}^2$

# statistical intervals

- interval estimates come with confidence levels
- $Z=\frac{\bar{X}-\mu}{\sigma / \sqrt{n}}$
- For p not close to 0.5, use Wilson score confidence interval (has extra terms)
- confidence interval - If multiple samples of trained typists were selected and an interval constructed for each sample mean, 95 percent of these intervals contain the true preferred keyboard height

# Tests on Hypotheses

- Var($\bar{X}-\bar{Y})=\frac{\sigma_1^2}{m}+\frac{\sigma_2^2}{n}$
- tail refers to the side we reject (e.g. upper-tailed=$H_a:\theta>\theta_0$
- $\alpha$ - type 1 - reject $H_0$ but $H_0$ true
- $\beta$ - type 2 - fail to reject $H_0$ but $H_0$ false
- we try to make the null hypothesis a statement of equality
- upper-tailed - reject large values
- $\alpha$ is computed using the probability distribution of the test statistic when $H_0$ is true, whereas determination of b requires knowing the test statistic distribution when $H_0$ is false
- type 1 error usually more serious, pick $\alpha$ level, then constrain $\beta$
- can standardize values and test these instead
- P-value is the probability, calculated assuming that the null hypothesis is true, of obtaining a value of the test statistic at least as contradictory to $H_0$ as the value calculated from the available sample. (observed significance level)
- reject $H_0$ if p $\leq \alpha$

# Inferences Based on 2 Samples

- $\sigma_{\bar{X}-\bar{Y}} = \sqrt{\frac{\sigma_1^2}{m}+\frac{\sigma_2^2}{n}}$
- there are formulas for type 1,2 errors
- If both normal, $Z = \frac{\bar{X}-\bar{Y}-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma_1^2}{m}+\frac{\sigma_2^2}{n}}}$
- If both have same variance, do a weighted average (pooled) $S_p^2 = \frac{m-1}{m+n-2}S_1^2+\frac{n-1}{m+n-2}S_2^2$
- If we have a large sample size, these expressions are basically true, we just use the sample standard deviation
- randomized controlled experiment - investigators assign subjects to the two treatments in a random fashion
- small sample sizes - two-sample t test
- $T = \frac{\bar{X}-\bar{Y}-(\mu_1-\mu_2)}{\sqrt{\frac{S_1^2}{m}+\frac{S_2^2}{n}}}$
- $\nu= \frac{(se_1^2 + se_2^2)^2}{\frac{se_1^4}{m-1}+\frac{se_2^4}{n-1}}$ (round down)
- $se_1 = \frac{s_1}{\sqrt{m}}$
- $se_2 = \frac{s_2}{\sqrt{n}}$
- two-sample t confidence interval for $\mu_1-\mu_2$ with confidence 100(1-a) percent:
- $\bar{x}-\bar{y} \pm t_{\alpha/2,v} \sqrt{\frac{s_1^2}{m}+\frac{s_2^2}{n}}$
- very hard to calculate type II errors here
- paired data - not independent
- we do a one-sample t test on the differences
- do pairing when large correlation within experimental units
- do independent-samples when correlation within pairs is not large
- proportions when m and n both large:
- $Z=\frac{\hat{p_1}-\hat{p_2}}{\sqrt{\hat{p}\hat{q}(\frac{1}{m}+\frac{1}{n})}}$ where $\hat{p}=\frac{m}{m+n}\hat{p_1}+\frac{n}{m+n}\hat{p_2}$, $\hat{q}=1-\hat{p}$
- bootstrap - computationally compute by taking samples, can use percentile intervals (sort and then pick nth from bottom/top)
- permutation tests - permute the labels on the data - p-value is the fraction of arrangements that are at least as extreme as the value computed for the original data
- for testing if two variances are equal, use $F_{\alpha,m-1,n-1}$
# ANOVA
- ANOVA - analysis of variance

### Regression and Correlations
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