---
layout: notes
section-type: notes
title: Linear Models
category: stat
---

* TOC
{:toc}

# ch 1 introduction
- regression analysis studies relationships among variables
- $Y = f(X_1,...X_i) + \epsilon$
- We can use $X_i^2$ as a term in a linear regression, but the function must be a linear combination of terms (no coefficients in the exponent, in a sine, etc.)
- regression analysis
	- statement of the probelm
	- select potential relevant variables
	- data collection
	- model specification
	- choice of fitting method
	- model fitting
	- model validation - important
	- iterative process!
- *regressions*
	- simple linear regression - univariate Y, univariate X
	- multiple linear regression - univariate Y, multivariate X
	- multivariate linear regression - multivariate Y
	- generalized linear regression - Y isn't normally distributed
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
- Gauss-Markov Theorem	- the least squares estimators $\hat{\beta_0}$ and $\hat{\beta_1}$ are unbiased estimators and have minimum variance among all unbiased linear estimators = best linear unbiased estimators.
	- unbiased means $E[\hat{x}] = E[x]$
	- $Var(\hat{\beta_1})=\frac{\sigma^2}{\sum(x_i-\bar{x})^2}$
	- $Var(\hat{\beta_0})=\frac{\sigma^2}{n}+\frac{(\bar{x}\sigma)^2}{\sum(x_i-\bar{x})^2}$
	- $\hat{\sigma}^2 = MSE = \frac{SSE}{n-2}$ - n-2 since there are 2 parameters in the linear model
- sometimes we have to enforce $\beta_0=0$, there are different statistics for this
		
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
- last column is P(>\\\|t\\\|)		
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
		- *PRESS*
		
# ch 8 automated search procedures

# ch 9 model building process overview
