---
layout: notes
section-type: notes
title: Freedman
category: stat
---
* TOC
{:toc}

# freedman
## ch 1 - causality
- When using observational (non-experimental) data to make causal inferences, the key problem is *confounding*
	- *stratification* = *cross-tabulation* - only look at when confounding variables have same value
- Generally, association is circumstantial evidence for causation
- examples
	- HIP trial of mammography - want to do whole treatment group v whole control group
	- Snow on cholera - water
	- causes of poverty - Yul's model, changes with lots of things

## ch 2 - the regression line
- regression line
	- goes through $(\bar{x}, \bar{y})$
	- slope: $r s_y / s_x$
	- intercept: $\bar{y} - slope \cdot \bar{x}$
	- basically fits graph of averages (minimizes MSE)
- SD line
	- same except slope: $sign(r) s_y / s_x$
	- intercept changes accordingly
- for regression, MSE = $(1-r^2) Var(Y)$

## ch 3 - matrix algebra
- adjoint - compute with mini-dets
- $A^{-1} = adj(A) / det(A)$
- PSD
	1. symmetric
	2. $x^TAx \geq 0$
- alternatively PSD iff $\exists$ diagonal D, orthogonal R s.t. $A=RDR^T$
- CLT
	- define $S_n = X_1 + ... + X_n$
	- define $Z_n = \frac{S_n - n \mu}{\sigma \sqrt{n}}$
	- $P(\|Z_n\| < 1) \to \int_{-1}^1 x ~ N(0, 1)$

## ch 4 - multiple regression
- multiple x, single y
- assumptions
	1. assume $n > p$ and X has full rank (rank p - columns are linearly independent)
	2. $\epsilon_i$ are iid, mean 0, variance $\sigma^2$
	3. $\epsilon$ independent of $X$
		- $e_i$ still orthogonal to $X$
- OLS is conditionally unbiased
	- $E[\hat{\theta} \| X] = \theta$
- $Cov(\hat{\theta}\|X) = \sigma^2 (X^TX)^{-1}$
	- $\hat{\sigma^2} = \frac{1}{n-p} \sum_i e_i^2$
		- this is unbiased - just dividing by n is too small since we have minimized $e_i$ so their variance is lower than var of $\epsilon_i$
- *random errors* $\epsilon$
- *residuals* $e$
- $H = X(X^TX)^{-1} X^T$
	1. e = (I-H)Y = $(I-H) \epsilon$
	2. H is symmetric
	3. $H^2 = H, (I-H)^2 = I-H$
	4. HX = X
	5. $e \perp X$
- basically H projects Y int R(X)
- $E[\hat{\sigma^2}\|X] = \sigma^2$
- random errs don't need to be normal
- variance
	- $var(Y) = var(X \hat{\theta}) + var(e)$
		- $var(X \hat{\theta})$ is the *explained variance*
		- *fraction of variance explained*: $R^2 = var(X \hat{\theta}) / var(Y)$
		- like summing squares by projecting
	- if there is no intercept in a regression eq, $R^2 = \|\|\hat{Y}\|\|^2 / \|\|Y\|\|^2$
- notes
	- if everything is integrable, OLS is unconditionally unbiased
	
## ch 5 - multiple regression - special topics
- drop assumption: $\epsilon$ independent of $X$
	- instead: $E[\epsilon\|X]=0, cov[\epsilon\|X] = \sigma^2 I$
	- can rewrite: $E[\epsilon]=0, cov[\epsilon] = \sigma^2 I$ fixing X
- *Gauss-markov thm* - assume linear model and assumption above: when X is fixed, OLS estimator is *BLUE* = best linear unbiased estimator
	- has smallest variance.
	- ***prove this***
- *generalized least squares regression model*: instead of above assumption, use $E[\epsilon\|X]=0, cov[\epsilon\|X] = G, \: G \in S^K_{++}$
	- covariance formula changes: $cov(\hat{\theta}_{OLS}\|X) = (X^TX)^{-1} X^TGX(X^TX)^{-1}$
	- estimator is the same, but is no longer BLUE - can correct for this:
		$(G^{-1/2}Y) = (G^{-1/2}X)\theta + (G^{-1/2}\epsilon)$
- *feasible GLS*=*Aitken estimator* - use $\hat{G}$
- examples
	- simple
	- iteratively reweighted
- *homoscedasticity*: $var(Y_i\|X)$ is the same for all i
	- opposite of *heteroscedasticity*
- 3 assumptions can break down:
	1. if $E[\epsilon\|X] \neq 0$ - GLS estimator is biased
	2. else if $cov(\epsilon\|X) \neq G$ - GLS unbiased, but covariance formula breaks down
	3. if G from data, but violates estimation procedure, estimator will be misealding estimate of cov
- ***skipped some pfs***

### 5.7 - normal theory
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
- *data snooping* - decide which hypotheses to test after examining data

## ch 6 - path models
- *path model* - graphical way to represent a regression equation
- making causal inferences by regression requires a *response schedule*

## ch 8 - bootstrap

## ch 9 - simultaneous equations
- *simultaneous-equation* models - use *instruemtanl variables / two-stage least squares*
	- these techniques avoid *simultaneity bias = endogeneity bias*ch

## ch 10 - issues in statistical modeling

# causal inference
- 2 general approaches
	1. matching - find patients that are similar and differ only in the treatment
	2. regression
		- requires *unconfoundedness* = *omitted variable bias*
		- if there are no confounders, correlation is causation
- Hainmueller & Hangartner (2013) - Swiss passport
	- naturalization decisions vary with immigrants' attributes
	- is there immigration against immigrants based on country of origin?
	- citizenship requires voting by municipality
- Sekhon et al. - when natural experiments are neither natural nor experiments
	- even when natural interventions are randomly as- signed, some of the treatment–control comparisons made available by natural experiments may not be valid
- Grossman et al. - "Descriptive Representation and Judicial Outcomes in Multiethnic Societies"
	- judicial outcomes of arabs depended on whether there was an Arab judge on the panel
- liver transplant
	- maximize benefit (life with - life without)
	- currently just goes to person who would die quickest without
	- Y = T Y(1) + (1-T) Y(0)
		- Y(1) = survival with transplant
		- Y(0) = survival w/out transplant
			- fundamental problem of causal inference - can't observe Y(1) and Y(0)
		- T = 1 if receive transplant else 0
	- goal: estimate $\tau = Y(1) - Y(0)$ for each person