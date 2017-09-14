---
layout: notes
section-type: notes
title: Regression
category: ai
---

[toc]

# problem formulation
- absorb intercept into feature vector of 1
	- x = column
	- add $x^{(0)} = 1$ as the first element
- matrix formation
	- $\hat{y} = f(x) = x^T \theta = \theta x^T = \theta_0 + \theta_1 x^1 + \theta_2 x^2 + ...$
	- $\pmb{x_1}$ is all the features for one data sample
	- $\pmb{x^1}$ is the first feature over all the data samples
	- our goal is to pick the optimal theta to minimize least squares
- loss function - minimize SSE
- SSE is a *convex function*
	- single point with 0 derivative
	- second derivative always positive
	- hessian is psd (*positive semi-definite*)

# optimization
- gradient - *denominator layout* - size of variable you are taking	- we always use denominator layout
	- *numerator layout* - you transpose the size
- *optimization* - find values of variables that minimize objective function while satisfying constraints

1. normal equations
	- $L(\theta) = \frac{1}{2} \sum_{i=1}^n (f(x_i)-y_i)^2$
	- = $1/2 (X \theta - y)^T (X \theta -y)$
	- set derivative equal to 0 and solve
	- $\theta = (X^TX)^{-1} X^Ty$
	- solving normal function is computationally expensive - that's why we do things like regularization (matrix multiplication is $O(n^3)$)
2. gradient descent = *batch gradient descent*
	- *gradient* - vector that points to direction of maximum increase
	- at every step, subtract gradient multiplied by learning rate: $x_k = x_{k-1} - \alpha \nabla_x F(x_{k-1})$
	- alpha = 0.05 seems to work
	- $J(\theta) = 1/2 (\theta ^T X^T X \theta - 2 \theta^T X^T y + y^T y)$
	- $\nabla_\theta J(\theta) = X^T X \theta - X^T Y$
		- = $\sum_i  x_i  (x_i^T - y_i)$
		- this represents residuals * examples
3. stochastic gradient descent
	- don't use all training examples - approximates gradient
		- single-sample
		- mini-batch (usually better in offline case)
	- *coordinate-descent* algorithm
	- *online* algorithm - update theta while training data is changing
	- when to stop?
		- predetermined number of iterations
		- stop when improvement drops below a threshold
	- each pass of the whole data = 1 epoch
	- benefits
		1. less prone to getting stuck to shallow local minima
		2. don't need huge ram
		3. faster
4. newton's method for optimization
	- *second-order* optimization - requires 1st & 2nd derivatives
	- $\theta_{k+1} = \theta_k - H_K^{-1} g_k$
	- update with inverse of Hessian as alpha - this is an approximation to a taylor series
	- finding inverse of Hessian can be hard / expensive

# evaluation
- accuracy = number of correct classifications / total number of test cases
- you train by lowering SSE or MSE on training data
	- report MSE for test samples
- *cross validation* - don't have enough data for a test set
	- data is reused
	- k-fold - split data into N pieces
		- N-1 pieces for fit model, 1 for test
		- cycle through all N cases
		- average the values we get for testing
	- leave one out (LOOCV)
		- train on all the data and only test on one
		- then cycle through everything
- *regularization path* of a regression - plot each coeff v. $\lambda$
	- tells you which features get pushed to 0 and when
	
# 1 - simple LR
- ml: task -> representation -> score function -> optimization -> models
	- all of these things are assumptions

# 2 - LR with non-linear basis functions
- can have nonlinear basis functions (ex. polynomial regression)
- radial basis function - ex. kernel function (Gaussian RBF)
	- $exp(-(x-r)^2 /  (2 \lambda ^2))$
- non-parametric algorithm - don't get any parameters theta; must keep data

# 3 - locally weighted LR
- recompute model for each target point
- instead of minimizing SSE, we minimize SSE weighted by each observation's closeness to the sample we want to query

# 4 - linear regression model with regularizations
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
	- argmin $\sum_i (y_i - \hat{y_i})^2+ \lambda ||\beta||_2^2 $
	- equivalent to minimizing $\sum_i (y_i - \hat{y_i})^2$ s.t. $\sum_j \beta_j^2 \leq t$
	- solution is $\hat{\beta_\lambda} = (X^TX+\lambda I)^{-1} X^T y$
	- for small $\lambda$ numerical solution is unstable
	- When $X^TX=I$, $\beta _{Ridge} = \frac{1}{1+\lambda} \beta_{Least Squares}$
2. *lasso* regression (L1)
	- $\sum_i (y_i - \hat{y_i})^2+\lambda  ||\beta||_1 $ 
	- equivalent to minimizing $\sum_i (y_i - \hat{y_i})^2$ s.t. $\sum_j |\beta_j| \leq t$
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
	- $\beta_{Naive ENet} = \sum_i (y_i - \hat{y_i})^2+\lambda_1 ||\beta||_1 + \lambda_2  ||\beta||_2^2$ 
	- l1 part generates sparse model
	- l2 part encourages grouping effect, stabilizes l1 regularization path
		- grouping effect - group of highly correlated features should all be selected
	- naive elastic net has too much shrinkage so we scale $\beta_{ENet} = (1+\lambda_2) \beta_{NaiveENet}$
	- to solve, fix l2 and solve lasso