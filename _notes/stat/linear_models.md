---
layout: notes
section-type: notes
title: linear models
category: stat
---

[TOC]

# ch 1 introduction

- regression analysis studies relationships among variables
- $Y = f(X_1,...X_i) + \epsilon$
- terms can be non-linear in X, but must be linear function of weights
- *regressions*
  - *simple linear regression* - univariate Y, univariate X
  - *multiple linear regression* - univariate Y, multivariate X
  - *multivariate linear regression* - multivariate Y
  - *generalized linear regression*s - Y isn't normally distributed
  - ANOVA - all X are categorical
  - Analysis of covariance - part of X are categorical

# ch 2 simple linear regression

### basics

- $Y = \beta_0 + \beta_1X + \epsilon$
- Take samples $x_i,y_i$
  - assume error $\epsilon \sim N(0,\sigma^2)$
  - further assume error $\epsilon_i,\epsilon_j,...\epsilon_n$ are i.i.d 
    - this isn't always the case, for example if some the data points were correlated to each other
- given $x_i$
  - $Var[Y_i] = Var[\epsilon_i] = \sigma^2$
  - $Y_i \sim N(\beta_0 + \beta_1X , \sigma^2)$
    - $Cov(Y_i,Y_j) = 0$, uncorrelated
- p-value - probability we reject $H_o$, but it is true
  - want this to be low to reject

### parameter estimation (least squares)

- $\epsilon_i = y_i - \beta_0 - \beta_1x_i$
- minimize sum of squared errors 
- Sums
  - SSE = $\sum_1^n\epsilon_i^2 = \sum_1^n (y_i - \beta_0 - \beta_1x_i)^2$
  - $S_{xx}=\sum (x_i-\bar{x})^2$
  - $S_{yy}=\sum (y_i-\bar{y})^2$
  - $S_{xy}=\sum (x_i-\bar{x})(y_i-\bar{y})$
- estimators
  - $\hat{\beta_1}=\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sum (x_i-\bar{x})^2}$
  - $\hat{\beta_0}=\bar{y}-\hat{\beta_1}\bar{x}$
  - Gauss-Markov Theorem- the least squares estimators $\hat{\beta_0}$ and $\hat{\beta_1}$ are unbiased estimators and have minimum variance among all unbiased linear estimators = best linear unbiased estimators.
  - unbiased means $E[\hat{x}] = E[x]$
  - $Var(\hat{\beta_1})=\frac{\sigma^2}{\sum(x_i-\bar{x})^2}$
  - $Var(\hat{\beta_0})=\frac{\sigma^2}{n}+\frac{(\bar{x}\sigma)^2}{\sum(x_i-\bar{x})^2}$
  - $\hat{\sigma}^2 = MSE = \frac{SSE}{n-2}$ - n-2 since there are 2 parameters in the linear model
- sometimes we have to enforce $\beta_0=0$, there are different statistics for this
  ​	

### evaluate model fitting

- SST - total sum of squares - measure of total variation in response variable
  - $\sum(y_i-\bar{y})^2$
- SSR - regression sum of squares - measure of variation explained by predictors
  - $\sum(\hat{y_i}-\bar{y})^2$
- SSE - measure of variation not explained by predictors
  - $\sum(y_i-\hat{y_i})^2$
- SST = SSR + SSE
- $R^2 = \frac{SSR}{SST}$ - coefficient of determination
  - measures the proportion of variation in Y that is explained by the predictor
- Cor(X,Y) = $\rho$ = $\frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}}$
  - only measures linear relationship
  - measure of strength and direction pof the linear association between two variables
  - better for simple linear regression, doesn't work later
  - $\rho^2$ = $R^2$

### inference for simple linear regression

- confidence interval construction
  - confidence interval (CI) is range of values likely to include true value of a parameter of interest
  - confidence level (CL) - probability that the procedure used to determine CI will provide an interval that covers the value of the parameter
- $\hat{\beta_0} \pm t_{n-2,\alpha /2} * s.e.(\hat{\beta_0}) $
  - for $\beta_1$
    - with known $\sigma$
      - $\frac{\hat{\beta_1}-\beta_1}{\sigma(\hat{\beta_1})} \sim N(0,1)$
      - derive CI
    - with unknown $\sigma$
      - $\frac{\hat{\beta_1}-\beta_1}{s(\hat{\beta_1})} \sim t_{n-2}$
      - derive CI
- hypothesis testing
  - t-test
    - $H_0:\beta_1=b $
    - $t_1 = \frac{\hat{\beta_1}-b}{s.e.(\hat{\beta_1})}$, n-2 degrees of freedom
  - f-test: $H_0:\beta_1=0 $
    - F=MSR/MSE  
    - reject if F > $F_{1-\alpha;1,n-2}$
- two kinds of prediction
  1. the prediction of the value of the respone variable Y which corresponds to any chose value, $x_o$ of the predictor variable
  2. the estimation of the mean response $\mu_o$ when X = $x_o$

### assumptions

1. There exists a linear relation between the response and predictor variable(s).
   - otherwise predicted values will be biased
2. The error terms have the constant variance, usually denoted as $\sigma^2$.
   - otherwise prediction / confidence intervals for Y will be affected
3. The error terms are independent, have mean 0.
   - otherwise a predictor like time might have been omitted from the model
4. Model fits all observations well (no outliers).
   - otherwise misleading fit
5. The errors follow a Normal distribution.
   - otherwise usually ok

- assessing regression assumptions

  1. look at scatterplot
  2. look at residual plot
     - should fall randomly near 0 with similar vertical variation, magnitudes
  3. Q-Q plot / normal probability plot
     - standardized residulas vs. normal scores
     - values should fall near line y = x, which represents normal distribution
  4. could do histogram of residuals
     - look for normal curve - only works with a lot of data points

  - lack of fit test - based on repeated Y values at same X values

### variable transformations

- if assumptions don't work, sometimes we can transform data so they work
- *transform x* - if residuals generally normal and have constant variance 
  - *corrects nonlinearity*
- *transform y* - if relationship generally linear, but non-constant error variance
  - *stabilizes variance*
- if both problems, try y first
- Box-Cox: Y' = $Y^l$ if l ≠ 0, else log(Y)

# ch 3 multiple linear regression

- $Y=\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ...+ \beta_p X_p + \epsilon$
  - each coefficient is the contribution of x_i after both variables have been linearly adjusted for the other predictor variables
- least squares solved to estimate regression coefficients
- unbiased $\hat{\sigma}^2 = \frac{SSE}{n-p-1}$

### matrix form

- write Y = X*$\beta+\epsilon$
  - each row is one $X_0,X_1,...,X_p$
- $\hat{\underline{\beta}} = (X'X)^{-1}X'Y$ 
  - multicollinearity - sometimes no unique soln - parameter estimates have large variability
- $\hat{\sigma}^2 = \frac{SSE}{n-p-1}$ where there are p predictors
- hat matrix - $\hat{Y}=HY$
  - H = $X(X'X)^{-1}X'$

### F-tests

- F statistic tests $H_0$: $\beta_1 = ... = \beta_p = 0$
  - reject when p ≤ .05
- $R^2$ only gets larger
  - adjusted $R^2$ - divide each sum of squares by its degrees of freedom
- partial f-test: $H_0$: $\beta_2 = \beta_3 = 0$ ~ any subset of betas is 0
  - tries to eliminate these from the model
  - can only eliminate 1 coefficient at a time
- extra sum of squares
  - first variables given go into model first
  - basically partial f-test, but calculate f in different way
    $H_0$: $\beta_2 = \beta_3 = 0$ ~ any subset of betas is 0
    - tries to eliminate these from the model

### anova table

- last column is $P(> \|t \|)$
- test is whether the statistic = 0
- default F-value is for all coefficients=0

### extra sums of squares

- regression happens in order you specify

# ch 4 multicollinearity

- *multicollinearity* - when predictors are highly correlated with each other
- roundoff errors
  1. X'X has determintant close to zero
  2. X'X elements differ substantially in magnitude
     - correlation transformation - normalizes variables
- when linearly dependent, clearly can't determine coefficients uniquely
- cannot interpret one set of regression coefficients as reflecting effects of the different predictors
- cannot extrapolate
- predicting is fine
- variance inflation factor (VIF) - measure how much the variances of the estimated regression coefficients are inflated as compared to when the predictors are not linearly related

# ch 5 categorical predictors

- quantitive variable - gets a number
- qualitative variable - ex. color
- A matrix is singular if and only if its determinant is zero
- *bonferroni procedure* - we are doing 3 tests with 5% confidence, so we actually do 5/3% for each test in order to restrict everything to 5% total

### indicator variables with 2 classes

- *ancova* - at least one categorical and one quantitative predictor
- indicator variables take on the value 0 or 1
  - *dummy coding* - matrix is singular so we drop the last indicator variable - called *reference* class / *baseline* class
- *additive effects* assume that each predictor’s effect on the response does not depend on the value of the other predictor (as long as the other one was fixed
  - assume they have the same slope
- *interaction effects* allow the effect of one predictor on the response to depend on the values of other predictors.
  - $y_i = β_0 + β_1x_{i1} + β_2xi2 + β_3xi1xi2 + ε_i$
- We can use the levene.test() function, from the lawstat package. The null hypothesis for this test is that the variances are equal across all classes.

### more than 2 classes

- $β_0$ is the mean response for the reference class when all the other predictors X1, X2, · · · are zero.
- $β_1$ is the mean response for the first class of C minus the mean response for the reference class when X1, X2, · · · are held constant.
- The F statistic reported with the summary() function for a linear model tests if β1 = ···β7 = 0.

## other coding

- effect coding
  - one vector is all -1s
  - B_0 should be weighted average of the class averages
- orthogonal coding *(not on test)*

# ch 6 polynomial regression

- have to get all lower order terms
- beware of overfitting
- must center all the variables to reduce multicollinearity
- hierarchical appraoch - fit higher order model to check whether lower order model is adequate or not
  - if a given order term is retained, all related terms of lower order must be retianed
  - otherwise it isn't invariant to transformations of the columns
- interaction terms are similar to before

# ch 7 model comparison and selection

- Ockham’s razor - principle of parsimony - given two theories that describe a phenomenon equally well, we should prefer the theory that is simpler
- several different criteria 
  - don't penalize many predictors
    - *$R^2_p$* - doesn't pen
  - penalize many predictors
    - *adjusted $R^2_p$* - penalty 
    - *Mallow's $C_p$*
    - *$AIC_p$*
    - *$BIC_p$*
    - *PRESS

# problem formulation

- absorb intercept into feature vector of 1
  - x = column
  - add $x^{(0)} = 1$ as the first element
- matrix formation
  - $\hat{y} = f(x) = x^T \theta = \theta x^T = \theta_0 + \theta_1 x^1 + \theta_2 x^2 + ...$
  - $\pmb{x_1}$ is all the features for one data sample
  - $\pmb{x^1}$ is the first fea\vert ture over all the data samples
  - our goal is to pick the optimal theta to minimize least squares

# normal equation

- gradient - *denominator layout* - size of variable you are taking	- we always use denominator layout
  - *numerator layout* - you transpose the size

- minimize: $L(\theta) = \|\|Y-Xw\|\|_2^2 \implies \hat{\theta} = (X^TX)^{-1} X^TY$

- 2 proofs

  1. set deriv and solve
  2. use projection matrix H to show HY is proj of Y onto R(X)

  - define projection matrix $H = X(X^TX)^{-1} X^T$
    - show $\|\|Y-X \theta\|\|^2 \geq \|\|Y - HY\|\|^2$
    - key idea: subtract and add HY


- solving normal function is computationally expensive - that's why we do things like regularization (matrix multiplication is $O(n^3)$)
- interpretation
  - if feature correlated, weights aren't stable / can't be interpreted
  - curvature inverse $(X^TX)^{-1}$ - dictates stability
- LS doesn't work when p >> n because of colinearity of X columns



# other notes

- normal lin reg
  - $y_i \sim N(\theta^Tx, \sigma^2)$
  - max likelihood for ***MSE***
- normal linear regression
  - variance MLE $\hat{\sigma}^2 = \sum (y_i - \hat{\theta}^T x_i)^2 / n$
    - in unbiased estimator, we divide by n-p
  - LS has a distr. $N(\theta, \sigma^2(X^TX)^{-1})​$
- linear regression model
  - when n is large, LS estimator ahs approx normal distr provided that X^TX /n is approx. PSD
- *confidence interval* - if we remade it 100 times, 95 would contain the true $\theta_1$
- type 1 err - like the tails of null distr.
  - *stat. significant*: p = 0.05
  - *highly stat. significant*: p = 0.01
- *weighted LS*: minimize $\sum [w_i (y_i - x_i^T \theta)]^2$
  - $\hat{\theta} = (X^TWX)^{-1} X^T W Y$
  - heteroscedastic normal lin reg model: erorrs ~ N(0, 1/w_i)
- *leverage scores* - measure how much each $x_i$ influences the LS fit
  - for data point i, $H_{ii}$ is the leverage score
- *LAD (least absolute deviation)* fit
  - MLE estimator when error is Laplacian



# LR with non-linear basis functions

- can have nonlinear basis functions (ex. polynomial regression)
- radial basis function - ex. kernel function (Gaussian RBF)
  - $exp(-(x-r)^2 /  (2 \lambda ^2))$
- non-parametric algorithm - don't get any parameters theta; must keep data

# locally weighted LR
- recompute model for each target point
- instead of minimizing SSE, we minimize SSE weighted by each observation's closeness to the sample we want to query

# linear regression model with regularizations

- when $(X^T X)$ isn't invertible can't use normal equations and gradient descent is likely unstable
  - X is nxp, usually n >> p and X almost always has rank p
  - problems when n < p
- intuitive way to fix this problem is to reduce p by getting read of features
- a lot of papers assume your data is already zero-centered
  - conventionally don't regularize the intercept term

### regularizations
1. *ridge* regression (L2)
  - if (X^T X) not invertible, add a small element to diagonal
  - then it becomes invertible
  - small lambda -> numerical solution is unstable
  - proof of why it's invertible is difficult
  - argmin $\sum_i (y_i - \hat{y_i})^2+ \lambda \vert \vert \beta\vert \vert _2^2 $
  - equivalent to minimizing $\sum_i (y_i - \hat{y_i})^2$ s.t. $\sum_j \beta_j^2 \leq t$
  - solution is $\hat{\beta_\lambda} = (X^TX+\lambda I)^{-1} X^T y$
  - for small $\lambda$ numerical solution is unstable
  - When $X^TX=I$, $\beta _{Ridge} = \frac{1}{1+\lambda} \beta_{Least Squares}$
2. *lasso* regression (L1)
  - $\sum_i (y_i - \hat{y_i})^2+\lambda  \vert \vert \beta\vert \vert _1 $ 
  - equivalent to minimizing $\sum_i (y_i - \hat{y_i})^2$ s.t. $\sum_j \vert \beta_j\vert  \leq t$
  - "least absolute shrinkage and selection operator"
  - lasso - least absolute shrinkage and selection operator - L1
  - acts in a nonlinear manner on the outcome y
  - keep the same SSE loss function, but add constraint of L1 norm
  - doesn't have closed form for Beta
    - because of the absolute value, gradient doesn't exist
    - can use directional derivatives
    - best solver is *LARS* - least angle regression
  - if tuning parameter is chose well, will set lots of coordinates to 0
  - convex functions / convex sets (like circle) are easier to solve
  - disadvantages
    - if p>n, lasso selects at most n variables
    - if pairwise correlations are very high, lasso only selects one variable
3. *elastic net* - hybrid of the other two
  - $\beta_{Naive ENet} = \sum_i (y_i - \hat{y_i})^2+\lambda_1 \vert \vert \beta\vert \vert _1 + \lambda_2  \vert \vert \beta\vert \vert _2^2$ 
  - l1 part generates sparse model
  - l2 part encourages grouping effect, stabilizes l1 regularization path
    - grouping effect - group of highly correlated features should all be selected
  - naive elastic net has too much shrinkage so we scale $\beta_{ENet} = (1+\lambda_2) \beta_{NaiveENet}$
  - to solve, fix l2 and solve lasso

# freedman

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