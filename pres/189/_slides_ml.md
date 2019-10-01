---
title: ml slides
separator: '----'
verticalSeparator: '---'
highlightTheme: ir-black
typora-copy-images-to: ./assets_files
---

<!-- .slide: data-transition="convex" data-transition-speed="medium"-->

<h1> machine learning </h1>
*press esc to navigate slides*

**<a href="https://twitter.com/chandan_singh96">@chandan_singh96</a>**

[![](assets_files/GitHub-Mark-64px.png)](https://github.com/csinva/csinva.github.io/blob/master/_slides/ml_slides/slides.md)



## <div> </div>

| Section               | Topic              |
| -- | -- |
| general | [intro](https://csinva.github.io/pres/189/#/1), [linear algebra](https://csinva.github.io/pres/189/#/2), [gaussian](https://csinva.github.io/pres/189/#/6), [parameter estimation](https://csinva.github.io/pres/189/#/4), [bias-variance](https://csinva.github.io/pres/189/#/5) |
| regression | [lin reg](https://csinva.github.io/pres/189/#/3), [LS](https://csinva.github.io/pres/189/#/7), [kernels](https://csinva.github.io/pres/189/#/9), [sparsity](https://csinva.github.io/pres/189/#/20) |
| dim reduction | [dim reduction](https://csinva.github.io/pres/189/#/8)|
| classification | [discr. vs. generative](https://csinva.github.io/pres/189/#/13), [nearest neighbor](https://csinva.github.io/pres/189/#/19), [DNNs](https://csinva.github.io/pres/189/#/12), [log. regression](https://csinva.github.io/pres/189/#/14), [lda/qda](https://csinva.github.io/pres/189/#/15), [decision trees](https://csinva.github.io/pres/189/#/21), [svms](https://csinva.github.io/pres/189/#/17) |
| optimization | [problems](https://csinva.github.io/pres/189/#/10), [algorithms](https://csinva.github.io/pres/189/#/11), [duality](https://csinva.github.io/pres/189/#/18), [boosting](https://csinva.github.io/pres/189/#/22), [em](https://csinva.github.io/pres/189/#/16) |



## pre-reqs

- linear algebra
- matrix calculus
- probability
- numpy/matplotlib

## reference

http://www.eecs189.org/



# introduction

## what's in an ml problem?

1. problems + data
2. model
3. loss function
4. optimization algorithm

## common ml problems

- regression
- classification
- density estimation
- dimensionality reduction

## types of problems <br>(by amount of labels)

- supervised
- ~semi-supervised
- reinforcement
- unsupervised

## types of models

- what problems do they solve?
- sort by how simple they are (bias/variance)
- discriminative vs. generative

## visual roadmap

![scikit_cheatsheet](assets_files/scikit_cheatsheet.png)

## parameters and hyperparameters

- result of optimization are a set of parameters
- parameters vs. hyperparameters

## train vs test

- training vs. testing error
- why do we need cross-validation?
- example: train, validate, test

# linear algebra review

## matrix properties

- *nonsingular* = invertible = nonzero determinant = null space of zero $\implies$ square
  - *rank* = dimension
  - *ill-conditioned matrix* - matrix is close to being singular - very small determinant

## vector norms

- $L_p-$norms: $||x||_p = (\sum_i \|x_i\|^p)^{1/p}$
- $||x||$ usually means $||x||_2$
  - $||x||^2 = x^Tx$


## matrix norms

- nuclear norm: $||X||_* = \sum_i \sigma_i$
- frobenius norm = euclidean norm (of a matrix): $||X||_F^2 =  \sqrt{\sum_i \sigma_i^2} $
- spectral norm = $L_2$ -norm (of a matrix) = $||X||_2 = \underset{\max}{\sigma}(X)$

## cauchy-shwartz inequality

$|x^T y| \leq ||x||_2 ||y||_2$

equivalent to the triangle inequality $||x+y||_2^2 \leq (||x||_2 + ||y||_2)^2$

## jensen's inequality

- $f(E[X]) \leq E[f(X)]$ for convex f

![1200px-ConvexFunction.svg](assets_files/1200px-ConvexFunction.svg.png)

## subspaces

![diagram0](assets_files/diagram0.svg)

## eigenvalues

- *eigenvalue eqn*: $Ax = \lambda x \implies (A-\lambda I)x=0$
  - $\det(A-\lambda I) = 0$ yields *characteristic polynomial*

## eigenvectors

![unequal](assets_files/unequal.png)

## eigenvectors in pca

![Screen Shot 2018-07-06 at 11.31.56 AM](assets_files/Screen Shot 2018-07-06 at 11.31.56 AM.png)

## evd

- *diagonalization* = *eigenvalue decomposition* = *spectral decomposition*
- assume A (nxn) is symmetric
  - $A = Q \Lambda Q^T$
  - Q := eigenvectors as columns, Q is orthonormal
  - $\Lambda$ diagonal

## evd extended

- only diagonalizable if n independent eigenvectors
- how does evd relate to invertibility?

![kulum-alin11-jan2014-28-638](assets_files/kulum-alin11-jan2014-28-638.jpg)

## svd
nxp matrix: $X=U \Sigma V^T$

- cols of U (nxn) are eigenvectors of $XX^T$
- cols of V (pxp) are eigenvectors of $X^TX$
- r singular values on diagonal of $\Sigma$ (nxp)
	- square roots of nonzero eigenvalues of both $XX^T$ and $X^TX$

## svd vs evd
- evd
  - not always orthogonal columns
  - complex eigenvalues
  - only square, not always possible
- svd
  - orthonormal columns
  - real/nonnegative eigenvalues
- symmetric matrices: eigenvalues real, eigenvectors orthogonal

## eigen stuff

- expressions when $A \in \mathbb{S}$
  - $\det(A) = \prod_i \lambda_i$
  - $tr(A) = \sum_i \lambda_i$
  - $\underset{\max}{\lambda}(A) = \sup_{x \neq 0} \frac{x^T A x}{x^T x}$
  - $\underset{\min}{\lambda}(A) = \inf_{x \neq 0} \frac{x^T A x}{x^T x}$

## positive semi-definite notation

- vectors: $x \preceq y$ means x is less than y elementwise
- matrices: $X \preceq Y$ means $Y-X$ is PSD
  - $v^TXv \leq v^TYv \:\: \forall v$

## psd

- defn 1: all eigenvalues are nonnegative
- defn 2: $x^TAx \geq 0 \:\forall x \in R^n$ 

## matrix calculus

- *gradient* vector $\nabla_x f(x)$- partial derivatives with respect to each element of function

## jacobian

function f: $\mathbb{R}^n \to \mathbb{R}^m$ 
*Jacobian matrix* : $$\mathbf J= \begin{bmatrix}    \dfrac{\partial \mathbf{f}}{\partial x_1} & \cdots & \dfrac{\partial \mathbf{f}}{\partial x_n} \end{bmatrix}$$

`$$= \begin{bmatrix}    \dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n}\\   \vdots & \ddots & \vdots\\    \dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n} \end{bmatrix}$$`

## hessian

function f: $\mathbb{R}^n \to \mathbb{R}$ 

`$$\mathbf H = \nabla^2 f(x)_{ij} = \frac{\partial^2 f(x)}{\partial x_i \partial x_j}$$` <div style="font-size: 23px;">
`$$= \begin{bmatrix}  \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1\,\partial x_n} \\[2.2ex]  \dfrac{\partial^2 f}{\partial x_2\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2\,\partial x_n} \\[2.2ex]  \vdots & \vdots & \ddots & \vdots \\[2.2ex]  \dfrac{\partial^2 f}{\partial x_n\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_n\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}\end{bmatrix}$$`
</div>

## nifty tricks

- `$x^TAx = tr(xx^TA) = \sum_{i, j} x_iA_{i, j} x_j$`
- $tr(AB)$ = sum of elementwise-products
- if X, Y symmetric, $tr(YX) = tr(Y \sum \lambda_i q_i q_i^T)$
- $A=UDV^T = \sum_i \sigma_i u_i v_i^T \implies A^{-1} = VD^{-1} U^T$

# linear regression

## regression
- what is regression?
- how does regression fit into the ml framework?

## feature engineering

- what is x?
- what is y?
- $\phi(x)$ can be treated like $x$

## lin. regression intuition 1

![1_yLeh6JjWHenfH4zFOA3HpQ](assets_files/1_yLeh6JjWHenfH4zFOA3HpQ.png)

## lin. regression setup

n = number of data points

d = dimension of each data point

`$$n\left\{\vphantom{\begin{bmatrix} X \\ \vdots \\. \end{bmatrix}}\right. \underbrace{ \begin{bmatrix} \vdots \\ y \\ \vdots \end{bmatrix}}_{\displaystyle 1}   = \underbrace{ \begin{bmatrix} \vdots \\ \cdots X \cdots\\ \vdots \end{bmatrix}}_{\displaystyle d} \vphantom{\begin{bmatrix} X \\ \vdots\\c \end{bmatrix}} \underbrace{ \begin{bmatrix} \vdots \\ w \\ \vdots \end{bmatrix}}_{\displaystyle 1} \left.\vphantom{\begin{bmatrix} X \\ \vdots \\. \end{bmatrix}}\right\}n$$`

## lin. regression intuition 2

![1_GT_lYlpF9e252-Rf6aQepw](assets_files/1_GT_lYlpF9e252-Rf6aQepw.jpeg)

## regularization

$\mathbf{\hat{y}} = \mathbf{X} \mathbf{\hat{w}}^T$

  

| Model | Loss |
| -- | -- |
|  OLS     | $\vert \vert y - \hat{y} \vert \vert^2$ |
| Ridge | $\vert \vert y - \hat{y} \vert \vert^2 + \lambda \vert\vert \hat w\vert\vert_2^2$ |
| Lasso | $\vert \vert y - \hat{y} \vert \vert^2 + \lambda \vert\vert \hat w\vert\vert_1$ |
| Elastic Net | $\vert \vert y - \hat{y} \vert \vert^2 + \lambda_1 \vert\vert \hat w\vert\vert_1+ \lambda_2\vert\vert \hat w\vert\vert_2^2$ |

## ols solution

- $\hat{w}_{OLS} = (X^TX)^{-1}X^Ty$
- 2 derivations: least squares, orthogonal projection

## ridge regression intuition

$\hat w_{RIDGE} = (X^TX \color{red}{+ \lambda I})^{-1}X^Ty$
![1d8XV](assets_files/1d8XV.png)



## later models will be nonlinear

<img src="assets_files/Screen Shot 2019-06-11 at 10.55.01 AM.png" width="80%">



# parameter estimation

## probabilistic model

- assume a **true underlying model**
- ex. $Y_i \sim \mathcal N(\theta^TX_i, \sigma^2)$
- this is equivalent to $P(Y_i|X_i; \theta) = \mathcal N(\theta^TX_i, \sigma^2)$

## bayes rule

$\overbrace{p(\theta \vert x)}^{\text{posterior}} = \frac{\overbrace{p(x\vert\theta)}^{\text{likelihood}} \overbrace{p(\theta)}^{\text{prior}}}{p(x)}$
![bayes2](assets_files/bayes2.jpg)

## likelihood 

$\mathcal L = p(data | \theta)$~ product over all n examples
- $p(x|\theta)$?
- $p(y|x; \theta)$?
- $p(x, y | \theta)$?
- $\to$ depends on the problem + model

## mle - maximum likelihood estimation

- $\hat{\theta}_{MLE} = \underset{\theta}{argmax} \: \mathcal{L}$
- associated with *frequentist* school of thought

## how to do mle problems

- write likelihood (product of probabilities)
- usually take log to turn product into a sum
- take derivative and set to zero to maximize (assuming convexity)

## map - maximum a posteriori

- $\hat{\theta}_{MAP} = \underset{\theta}{argmax} \: p(\theta\vert x) = \underset{\theta}{argmax} \: p(x\vert \theta) p(\theta)  \\\ = \underset{\theta}{argmax} \: [ \log \: p(x\vert\theta) + \log \: p(\theta) ]$
  - $p(x)$ disappears because it doesn't depend on $\theta$
- associated with *bayesian* school of thought

## mle vs. map

- $\hat{\theta}_{MLE} = \underset{\theta}{argmax} \: \overbrace{p(x|\theta)}^{\text{likelihood}}$
- $\hat{\theta}_{MAP} = \underset{\theta}{argmax} \: \overbrace{p(\theta\vert x)}^{\text{posterior}} = \underset{\theta}{argmax} \: p(x\vert \theta) \color{cadetblue}{\overbrace{p(\theta)}^{\text{prior}}}$
  - ```$\hat{\theta}_{\text{Bayes}} = E_\theta \: p(\theta|x) $```

# bias-variance tradeoff

## intuition 1

![bias-and-variance](assets_files/bias-and-variance.jpg)

## intuition 2

![fittings](assets_files/fittings.jpg)

## bias

- bias of a model: $E[\hat{f}(x) - f(x)]$
  - expectation over drawing new training sets from same distr.
- could also have bias of a point estimate: $E[\hat{\theta} - \theta]$

## variance

- "estimation error"
- variance of a model: $V[\hat{f}(x)] = E\left[\big(\hat{f}(x) - E[\hat{f}(x)]\big)^2\right]$
  - expectation over training sets with fixed x

## bias-variance trade-off

- mean-squared error of model: $E[(\hat{f}(x) - f(x))^2]$
  - = bias$^2$ + variance
  - = $E[\hat{f}(x) - f(x)]^2$ + $E[(\hat{f(x)} - E[\hat{f(x)}])^2]$

![biasvariance](assets_files/biasvariance.png)

# multivariate gaussian

## definitions

- $p(x\vert\mu, \Sigma) = \frac{1}{(2\pi )^{n/2} \vert\Sigma\vert^{1/2}} \exp\left[ -\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right]$
  - $\mu$ is mean vector
  - $\Sigma$ is covariance matrix

![MultivariateNormal](assets_files/MultivariateNormal.png)

## understanding $\Sigma$

![main-qimg-988175a0da2cb7674391d43a2edab558](assets_files/main-qimg-988175a0da2cb7674391d43a2edab558.png)

## understanding $\Sigma^{-1}$

![slide_11](assets_files/slide_11.jpg)

## mle gaussian estimation

- $\hat \mu,  \hat \Sigma = argmax \:  P(x_1, ..., x_n|\mu, \Sigma)$
- $\hat \mu = \frac{1}{n} \sum x_i$
- $\hat \Sigma = \frac{1}{n} \sum (x_ i - \hat \mu)(x_i - \hat \mu)^T$

# advanced linear least squares

## weighted least squares

- weight certain points more $\omega_i$
- $\hat{w}_\text{wls} = argmin \left( \sum \omega_i (y_i - x_i^T w)^2\right)$
- $= (X^T\Omega X)^{-1}X^T\Omega y$

## generalized least squares

- noise variables are not independent

![Heteroscedasticity](assets_files/Heteroscedasticity.jpg)

## overview

![Screen Shot 2018-06-24 at 7.40.57 PM](assets_files/Screen Shot 2018-06-24 at 7.40.57 PM.png)

## total LS intuition

- add i.i.d. gaussian noise in x and y - regularization

![Screen Shot 2018-06-29 at 4.08.09 PM](assets_files/Screen Shot 2018-06-29 at 4.08.09 PM.png)

## total LS solution

- $\hat{w}_{TLS} = (X^TX - \sigma^2 I)^{-1}X^Ty$
  - here, $\sigma$ is last singular value of $[X \: y]$

# dimensionality reduction

## pca intuition

orthogonal dimensions that maximize variance of $X$

![pca](assets_files/pca.png)

## pca in python

```python
X -= np.mean(X, axis=0) #zero-center data (nxd) 
cov = np.dot(X.T, X) / X.shape[0] #get cov. matrix (dxd) 
U, D, V = np.linalg.svd(cov) #compute svd, (all dxd) 
X_2d = np.dot(X, U[:, :2]) #project in 2d (nx2)
```

## pca in practice

- eigenvalue represents prop. of explained variance: $\sum \lambda_i = tr(\Sigma) = \sum Var(X_i)$	
- use svd
- adaptive PCA is faster (sequential)

## cca

- linearly independent dimensions that maximize correlation between $X, Y$
- invariant to scalings / affine transformations of X, Y

  ![cca](assets_files/cca.jpg)

## correlations

![correlations](assets_files/correlations.png)

# kernels

## why are kernels useful?

![data_2d_to_3d_hyperplane](assets_files/data_2d_to_3d_hyperplane.png)

## ex. ridge regression

- reformulate the problem to be computationally efficient + nonlinear
  - matrix inversion is ~$O(dim^3)$
- $\hat{w} = (\color{red}{\underbrace{X^TX}_{dxd}} + \lambda I)^{-1}X^Ty$ ~ faster when $\color{red}{d << n}$
- $\hat{w} = X^T(\color{red}{\underbrace{XX^T}_{nxn}} + \lambda I)^{-1}y$ ~ faster when $\color{red}{n << d}$

## kernels

![Screen Shot 2018-06-24 at 9.53.55 PM](assets_files/Screen Shot 2018-06-24 at 9.53.55 PM.png)

- $\phi_i^T\phi_j = \phi(x_i)^T \phi(x_j)$

## kernel trick ex.

- $\mathbf{x} = [x_1, x_2]$
- $\phi(\mathbf x) = \begin{bmatrix} x_1^2 & x_2^2 &\sqrt{2}x_1x_2 & \sqrt{2}x_1 & \sqrt{2}x_2 &1\end{bmatrix}^T $

`$k(\mathbf{x}, \mathbf{z}) = \underbrace{\phi(\mathbf x)^T \phi (\mathbf z)}_{\text{O(augmented feature space)}} = \underbrace{(\mathbf x^T \mathbf z+ 1)^2}_{\text{O(original feature space + log(degree))}}$`
- another ex. rbf kernel: $k(\mathbf x, \mathbf z) = \exp(-\gamma \vert \vert \mathbf x - \mathbf z \vert \vert ^2 )$


## different from kernel regression...

- note, what discussed here is different from the nonparametric technique of kernel regression: 
- ```$\widehat{y}_h(x)=\frac{\sum_{i=1}^n K_h(x-x_i) y_i}{\sum_{j=1}^nK_h(x-x_j)} $```
  - K is a kernel with a bandwidth h

# optimization problems

## overview

- minimizing things
- ex. $\underset{\theta}{arg min} \: \sum \big(y_i - f(x_i; \theta)\big)^2$

![loss_surfaces](assets_files/loss_surfaces.jpg)

## convexity

- Hessian $\nabla^2 f(x) \succeq 0 \: \forall x$

- $f(x_2) \geq f(x_1) + \nabla f(x_1) (x_2 - x_1)$

  ![ tangents](assets_files/ tangents.png)

## convexity continued

$\color{purple}{t f(x_1) + (1-t) f(x_2)} \geq f(tx_1 + (1-t)x_2)$

![1200px-ConvexFunction.svg](assets_files/1200px-ConvexFunction.svg.png)

## strong convexity + smoothness

$0 \preceq \underset{\text{strong convexity}}{mI} \preceq \nabla^2 f(x) \preceq \underset{\text{smoothness}}{MI}$

![bounds](assets_files/bounds.jpg)

## smoothness

M-smooth = Lipschitz continuous gradient: $||\nabla f(x_2) - \nabla f(x_1)|| \leq M||x_2-x_1||\quad \forall x_1,x_2$

| Lipschitz continuous f | M-smooth |
| :-- | --:- |
|  	![lipschitz_continuous_func](assets_files/lipschitz_continuous_func.gif) | ![lipschitz](assets_files/lipschitz.jpg)  |


# optimization algorithms

## gradient descent

![0_QwE8M4MupSdqA3M4](assets_files/0_QwE8M4MupSdqA3M4.png)

## when do we stop?

- validation error stops changing
- changes become small enough

## stochastic gradient descent

![stochastic-vs-batch-gradient-descent](assets_files/stochastic-vs-batch-gradient-descent.png)



## [momentum demo](https://distill.pub/2017/momentum/)

- $$\theta^{(t+1)} = \theta^{(t)} - \alpha_t \nabla f(\theta^{(t)}) + \color{cornflowerblue}{\underset{\text{momentum}}{\beta_t (f(\theta^{(t)}) - f(\theta^{(t-1)}))}}$$

  <div class="divmomentum">
      <iframe class="iframemomentum" src="https://distill.pub/2017/momentum/" scrolling="no" frameborder="no" style="position:absolute; top:-165px; left: -25px; width:1420px; height: 768px"></iframe>
  </div>


## newton-raphson

![slide_8](assets_files/slide_8.jpg)
- apply to find roots of **f'(x)**: $\theta^{(t+1)} = \theta^{(t)} - \nabla^2 f(\theta^{(t)})^{-1}\nabla f(\theta^{(t)})$

## gauss-newton

- modify newton's method assuming we are minimizing nonlinear least squares
- $\theta^{(t+1)} = \theta^{(t)} - \nabla^2 f(\theta^{(t)})^{-1}\nabla f(\theta^{(t)})$
- $\theta^{(t+1)} = \theta^{(t)} + \color{cadetblue}{(J^TJ)}^{-1} \color{cadetblue}{J^T\Delta y}$ $\quad J$ is the Jacobian

# neural nets

## so much hype

- it predicts: vision, audio, text, ~rl
- it's easy: little feature engineering
- it generalizes, despite having many parameters

## perceptron

![perceptron](assets_files/perceptron.png) ![perceptron-1](assets_files/perceptron-1.gif)

## training a perceptron

- loss function: $L(x, y; w) = (\hat{y} - y)^2$
- goal: $\frac{\partial L}{\partial w_i}$ for all weights
- calculate efficiently with backprop

## [backprop demo](https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/)

- [nn demo playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.63885&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)


## going deeper

![1_gccuMDV8fXjcvz1RSk4kgQ](assets_files/1_gccuMDV8fXjcvz1RSk4kgQ.png)



## coding DNNs in numpy

```python
from numpy import exp, array, random
X = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
Y = array([[0, 1, 1, 0]]).T
w = 2 * random.random((3, 1)) - 1
for iteration in range(10000):
    Yhat = 1 / (1 + exp(-(X @ w)))
    w += X.T @ (Y - Yhat) * Yhat * (1 - Yhat)
print(1 / (1 + exp(-(array([1, 0, 0] @ w))))
```



## coding DNNs in advanced numpy

```python
import tensorflow as tf
import torch
```

## cnns

![cnns](assets_files/cnns.gif)

## cnns 2

![cnn2](assets_files/cnn2.jpeg)

## rnns

![RNN-longtermdependencies](assets_files/RNN-longtermdependencies.png)

## connection to the brain?

![30124068_2022213811328985_1877822313844441088_o](assets_files/30124068_2022213811328985_1877822313844441088_o.png)

# discriminative vs. generative

## definitions

![discriminative_vs_generative](assets_files/discriminative_vs_generative.png)

- discriminative: $p(y|x)$
- generative: $p(x, y) = p(x|y) p(y)$

## sorting models

<div style='float:left;width:32%;' class='centered'>

<strong> generative </strong> </br>

bayes classifier </br>

hmms </br>

lda/qda </br>

</div>

<div style='float:right;width:32%;'>
<strong> discriminative </strong> </br>

linear regression </br>

svms </br>

nearest neighbor </br>

decision trees / random forests </br>

</div>

## bayes classifier

- risk: $\mathbb E_{(X,Y)}[L(f(x), y) ] = \sum_x p(x) \sum_y L(f(x), y) p(y|x)$
- bayes classifier: $f^*(x) = \underset{y}{argmin} \: \underset{y}{\sum} \:L(y, y') p(y'|x)$
  - given x, pick y that minimizes risk
- with 0-1 error: $f^*(x) = \underset{y}{argmax} \: p(y|x)$

## bayes classifier example

- with 0-1 error: $f^*(x) = \underset{y}{argmax} \: p(y|x) = \underset{y}{argmax} \: p(x|y) \cdot p(y)$
  - let y be sentiment (positive or negative)
  - let x be words 

# logistic regression

## definitions

![Exam_pass_logistic_curve](assets_files/Exam_pass_logistic_curve.jpeg)

- $\sigma(z) = \frac{1}{1+e^{-z}}$
- $P(\hat{Y}=1|x; w) = \sigma(w^Tx)$
  - threshold to predict
  - not really regression

## comparison with OLS

![VVtRW](assets_files/VVtRW.png)

![nEC4H](assets_files/nEC4H.png)

## loss functions

- log-loss = cross-entropy: $-\sum_x p(x) \: log \: q(x)$
  - $p(x)$ true $y$
  - $q(x)$ predicted probability of y
- corresponds to MLE for Bernoulli

![Screen Shot 2018-07-02 at 11.26.42 AM](assets_files/Screen Shot 2018-07-02 at 11.26.42 AM.png)

## multiclass

- one-hot encoding: $[1, 0, 0]$, $[0, 1, 0]$, $[0, 0, 1]$
- softmax function: $\sigma(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_j \exp z_j}$
- loss function still cross-entropy

![multihierarchy](assets_files/multihierarchy.png)

## training

- no closed form, but convex loss $\implies$ convex optimization!
  - minimize loss on cross-entropy (where p(x) is modelled by sigmoid)
  - or maximize likelihood

# gaussian discriminant analysis

## generative model

![comparisons](assets_files/comparisons.png)

## assumptions

- $\hat{y} = \underset{y}{\text{argmax}} \: p(y|\mathbf{x}) = \underset{y}{\text{argmax}} \: P(\mathbf{x}|y)p(y)$
  - $P(\mathbf{x}|y)\sim \mathcal N (\mathbf \mu_y, \Sigma_y)$: there are |Y| of these
  - $p(y) = \frac{n_y}{n}$: 1 of these

## lda vs. log. regression

- differences
  - generative
  - treats each class independently
- same
  - form for posterior (sigmoid / softmax)

## dimensionality reduction

![lda_1](assets_files/lda_1.png)



## multiclass lda vs. qda

![Screen Shot 2018-07-21 at 10.41.16 AM](assets_files/Screen Shot 2018-07-21 at 10.41.16 AM.png)

# em

## k-means (2d)

![k-means](assets_files/k-means.gif)

## mixture of gaussians (2d)



![](assets_files/inside-cluster-em.gif)![](assets_files/ad8e9b45e3d01deef10f0cc07ec22144c3c631b3.gif)



## mixture of gaussians (1d)

![mixture-iterations](assets_files/mixture-iterations.gif)

## EM

- want to maximize *complete log-likelihood* $l (\theta; x, z) = log \: p(x,z\|\theta)$ but don't know latent z
  - *expectation step* - values of z filled in
  - *maximization step* - parameters are adjusted based on z

## simplifying the math

- $x$: observed vars, $z$: latent vars, $q$: assignments to z
- E: $q^{(t+1)} (z|x) = \underset{q}{argmin} \: D(q||\theta^{(t)})$
  - lower bound on complete log-likelihood (pf: Jensen's inequality)
- M: $\theta^{(t+1)} = \underset{\theta}{argmin} \: D(q^{(t+1)} || \theta)$

# svms

[note 20](http://www.eecs189.org/static/notes/n20.pdf) is good

## perceptron/logistic reg. problems

![Screen Shot 2018-07-02 at 3.37.07 PM](assets_files/Screen Shot 2018-07-02 at 3.37.07 PM.png)

- doesn't find best solution
- unstable when data not linearly separable

## what's w?

```$\hat{y} =\begin{cases}   1 &\text{if } w^Tx +b \geq 0 \\ -1 &\text{otherwise}\end{cases}$```

![svm_w](assets_files/svm_w.png)

## how far are points?

decision boundary: {$x: w^Tx - b = 0$}

$D = \frac{|w^T(z-x_0)|}{||w||_2} = \frac{|w^Tz-b|}{||w||_2}$

![svm_proj](assets_files/svm_proj.png)

## hard margin intuition

```$\begin{align} \underset{m, w, b}{\max} \quad &m \\ s.t. \quad &y_i \frac{(w^Tx_i-b)}{||w||_2} \geq m \: \forall i\\ &m \geq 0\end{align}$```

## hard margin formulation

- let $m = 1 / ||w||_2 \implies$unique soln

```$\underset{w, b}{\min}\quad \frac{1}{2} ||w||_2^2 \\s.t. \quad y_i (w^Tx_i - b) \geq 1\: \forall i$```

## ${\color{cadetblue}{\text{soft}}}$ margin

```$\begin{align}\underset{w, b, \color{cadetblue}\xi}{\min}\quad &\frac{1}{2} ||w||_2^2 \color{cadetblue}{+C\sum_i \xi_i}\\s.t. \quad &y_i (w^Tx_i + b) \geq 1 \color{cadetblue}{-\xi_i}\: \forall i\\ &\color{cadetblue}{\xi_i \geq 0 \: \forall i}\end{align}$```

![errs](assets_files/errs.png)

## binary classification

can rewrite by absorbing $\xi$ constraints

$\underset{w, b}{\min} \quad \frac{1}{2}||w||^2 + C\sum_i \max(1-y_i(w^Tx_i - b), 0)$

## summarizing models

![losses](assets_files/losses.png)

- svm: hinge loss
- log. regression: log loss
- perceptron: perceptron loss



## binary classification

| Model               | $\mathbf{\hat{\theta}}$ objective (minimize)                 |
| -- | -- |
| Perceptron          | $\sum_i \max(0,  -y_i \cdot \theta^T x_i)$                   |
| Linear SVM          | $\theta^T\theta + C \sum_i \max(0,1-y_i \cdot \theta^T x_i)$ |
| Logistic regression | $\theta^T\theta + C \sum_i \log[1+\exp(-y_i \cdot \theta^T x_i)]$ |

# duality

## problem

<div style='float:left;width:30%;' class='centered'>

<h3> primal </h3>
```$p^* = \min \: f_0 (x) \\ s.t. \: f_i(x) \leq 0 \\ h_i(x) = 0$```

</div>

<div style='float:right;width:65%;'>

<h3> dual </h3>
```$d^* = \underset{\lambda, \nu}{\max} \: \overbrace{\underset{x}{\inf} \: \underbrace{f_0(x) + \sum \lambda_i f_i(x) + \sum \nu_i h_i(x)}_{\text{Lagrangian} \: L(x, \lambda, \nu)}}^{\text{dual function} \: g(\lambda, \nu)} \\s.t. \: \lambda \succeq 0\\$```

</div>

## comments

- *dual function* $g(\lambda, \nu)$ always concave
  - $\lambda \succeq 0 \implies g(\lambda, \nu) \leq p^*$

- $(\lambda, \nu)$ *dual feasible* if
  1. $\lambda \succeq 0$
  2. $(\lambda, \nu) \in dom \: g$

## duality

- *weak duality*: $d^\ast \leq p^*$

  - *optimal duality gap*: $p^\ast - d^*$

- *strong duality*: $d^\ast = p^\ast$ ~ requires more than convexity

# nearest neighbor

## intuition

![Screen Shot 2018-07-03 at 12.14.53 AM](assets_files/Screen Shot 2018-07-03 at 12.14.53 AM.png)

## comments

- no training, slow testing
- nonparametric: huge memory
- how to pick distance?
- poor in high-dimensions
- theoretical error rate

# sparsity

```$\underset{w}{\min} \quad ||Xw-y||_2^2\\s.t.\quad||w||_0 \leq k$```

## constrained form

<div style='float:left;width:45%;' class='centered'>

lasso: ```$\underset{w}{\min} \quad ||Xw-y||_2^2\\s.t.\quad\color{cadetblue}{||w||_1} \leq k$```

</div>

<div style='float:right;width:45%;'>

ridge: ```$\underset{w}{\min} \quad ||Xw-y||_2^2\\s.t.\quad\color{cadetblue}{||w||_2} \leq k$```

</div>


![Screen Shot 2018-07-03 at 9.16.55 AM](assets_files/Screen Shot 2018-07-03 at 9.16.55 AM.png)

## dual form

lasso: $\underset{w}{\min} \quad ||Xw-y||_2^2 + \color{cadetblue}{ \lambda ||w||_1}$

- $\color{cadetblue}{\Delta = \lambda}$

ridge: $\underset{w}{\min} \quad ||Xw-y||_2^2 + \color{cadetblue}{\lambda ||w||_2^2}$

- $\color{cadetblue}{\Delta = 2 \lambda w}$

## lasso optimization

- coordinate descent: requires jointly convex
  - closed form for each $w_i$
  - iterate, might re-update $w_i$

## matching pursuit

- start with all 0s
- iteratively choose/update $w_i$ to minimize $||y-Xw||^2$

![Screen Shot 2018-07-03 at 9.51.10 AM](assets_files/Screen Shot 2018-07-03 at 9.51.10 AM.png)

## orthogonal matching pursuit

- at each step, update all nonzero weights

# decision trees / random forests

## decision tree intuition

![decision](assets_files/decision.png)

## training

- greedy - use metric to pick attribute
  - split on this attribute then repeat
  - high variance

## information gain

maximize H(parent) - [weighted average] $\cdot$ H(children)

- often picks too many attributes

![c50](assets_files/c50.png)


## info theory

- maximize $I(X; Y) \equiv$ minimize $H(Y|X)$

![entropy-venn-diagram](assets_files/entropy-venn-diagram.png)

## split functions
- info gain (approximate w/ gini impurity)
- misclassification rate
- (40-40);  could be: (30-10, 10-30), (20-40, 20-0)

![errs-0983654](assets_files/errs-0983654.png)

## stopping

- depth
- metric
- node proportion
- pruning

## random forests

- multiple classifiers
- *bagging* = *bootstrap aggregating*: each classifier uses subset of datapoints
- *feature randomization*: each split uses subset of features

## random forest voting

- consensus
- average
- *adaboost*

## regression tree

- stop splitting at some point and apply linear regression

## other values

- missing values - fill in with most common val / probabilistically
- continuous values - split on thresholds

# boosting

- sequentially train many *weak learners* to approximate a function

## adaboost

- initialize weights to 1/n
- iterate
  - classify weighted points
  - re-weight points to emphasize errors
- finally, output error-weighted sum of weak learners

## adaboost comments

- derived using exponential loss risk minimization (freund and schapire)
- test error can keep decreasing once training error is at 0

## gradient boosting

- weak learners applied in sequence
- subtract gradient of loss with respect to current total model
  - for squared loss, just the residual
- models need not be differentiable

## xgboost

- very popular implementation of gradient boosting
- fast and efficient