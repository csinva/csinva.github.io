---
title: Chandan's slides
separator: '----'
verticalSeparator: '---'
theme: black
highlightTheme: ir-black
typora-copy-images-to: ./assets
---

<!-- .slide: data-transition="convex" data-transition-speed="medium"-->

<h1> machine learning </h1>
**chandan singh**

*press esc to navigate slides*

## <div> </div>

| Section               | Topic              |
| -- | -- |
| General | [intro](https://csinva.github.io/pres/189/#/1), [linear algebra](https://csinva.github.io/pres/189/#/2), [gaussian](https://csinva.github.io/pres/189/#/6), [parameter estimation](https://csinva.github.io/pres/189/#/4), [bias-variance](https://csinva.github.io/pres/189/#/5) |
| Regression | [lin reg](https://csinva.github.io/pres/189/#/3), [LS](https://csinva.github.io/pres/189/#/7), [kernels](https://csinva.github.io/pres/189/#/9), [sparsity](https://csinva.github.io/pres/189/#/20) |
| Dim reduction | [dim reduction](https://csinva.github.io/pres/189/#/8)|
| Classification | [discr. vs. generative](https://csinva.github.io/pres/189/#/13), [nearest neighbor](https://csinva.github.io/pres/189/#/19), [DNNs](https://csinva.github.io/pres/189/#/12), [log. regression](https://csinva.github.io/pres/189/#/14), [lda/qda](https://csinva.github.io/pres/189/#/15), [decision trees](https://csinva.github.io/pres/189/#/21), [svms](https://csinva.github.io/pres/189/#/17) |
| Optimization | [problems](https://csinva.github.io/pres/189/#/10), [algorithms](https://csinva.github.io/pres/189/#/11), [duality](https://csinva.github.io/pres/189/#/18), [boosting](https://csinva.github.io/pres/189/#/22), [em](https://csinva.github.io/pres/189/#/16) |



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

## types of ml problems

- regression
- classification
- density estimation
- dimensionality reduction

## types of problems (by amount of labels)

- supervised
- ~semi-supervised
- reinforcement
- unsupervised

## types of models

- sort by bias/variance
- discriminative vs. generative

## visual roadmap

![scikit_cheatsheet](assets/scikit_cheatsheet.png)

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

- $||x||$ is the $L_2$ norm
- $||x||^2 = x^Tx$
- $L_p-$norms: $||x||_p = (\sum_i \|x_i\|^p)^{1/p}$

## matrix norms

- *frobenius*: like $L_2$ for matrices
- *spectral norm* = $L_2$-norm of a matrix
  - $||X||_2 = \underset{\max}{\sigma}(X)$

## cauchy-shwartz inequality

- $|x^T y| \leq ||x||_2 ||y||_2$
- $\iff$ triangle inequality $||x+y||^2 \leq (||x|| + ||y||)^2$

## jensen's inequality

- $f(E[X]) \leq E[f(X)]$ for convex f

![1200px-ConvexFunction.svg](assets/1200px-ConvexFunction.svg.png)

## subspaces

![diagram0](assets/diagram0.svg)

## eigenvalues

- *eigenvalue eqn*: $Ax = \lambda x \implies (A-\lambda I)x=0$
  - $\det(A-\lambda I) = 0$ yields *characteristic polynomial*

## eigenvectors

![unequal](assets/unequal.png)

## eigenvectors in pca

![Screen Shot 2018-07-06 at 11.31.56 AM](assets/Screen Shot 2018-07-06 at 11.31.56 AM.png)

## evd

- *diagonalization* = *eigenvalue decomposition* = *spectral decomposition*
- assume A (nxn) is symmetric
  - $A = Q \Lambda Q^T$
  - Q := eigenvectors as columns, Q is orthonormal
  - $\Lambda$ diagonal

## evd extended

- only diagonalizable if n independent eigenvectors
- how does evd relate to invertibility?

![kulum-alin11-jan2014-28-638](assets/kulum-alin11-jan2014-28-638.jpg)

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
  - $\|\|A\|\|_2 = \max \| \lambda_i \|$
  - $\|\|A\|\|_F = \sqrt{\sum \lambda_i^2}$
  - $\underset{\max}{\lambda (A)} = \sup_{x \neq 0} \frac{x^T A x}{x^T x}$
  - $\underset{\min}{\lambda(A)} = \inf_{x \neq 0} \frac{x^T A x}{x^T x}$

## positive semi-definite notation

- vectors: $x \preceq y$ means x is less than y elementwise
- matrices: $X \preceq Y$ means $Y-X$ is PSD
  - $v^TXv \leq v^TYv \:\: \forall v$

## psd

- defn 1: all eigenvalues are nonnegative
- defn 2: $x^TAx \geq 0 \:\forall x \in R^n$ 

## matrix calculus

- *gradient* vector $\nabla_x f(x)$ - partial derivatives with respect to each element of function

## jacobian

function f: $\mathbb{R}^n \to \mathbb{R}^m$ 

![Screen Shot 2018-07-06 at 1.12.13 PM](assets/Screen Shot 2018-07-06 at 1.12.13 PM.png)

## hessian

function f: $\mathbb{R}^n \to \mathbb{R}$ 

![Screen Shot 2018-07-06 at 1.12.17 PM](assets/Screen Shot 2018-07-06 at 1.12.17 PM.png)

## nifty tricks

- $x^TAx = tr(xx^TA) = \sum_{i, j} x_iA_{i, j} x_j$
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

![1_yLeh6JjWHenfH4zFOA3HpQ](assets/1_yLeh6JjWHenfH4zFOA3HpQ.png)

## lin. regression intuition 2

- here, $\mathbf{y}$ is n-dimensional

![1_GT_lYlpF9e252-Rf6aQepw](assets/1_GT_lYlpF9e252-Rf6aQepw.jpeg)

## linear regression types

- $\hat{y} = \mathbf{w}^T \mathbf{x}$

  

  | Model | Loss |
  | -- | -- |
  |  OLS     | $\sum_i (y_i - \hat{y}_i)^2$ |
  | Ridge | $\sum_i (y_i - \hat{y_i})^2$ + $\lambda \vert\vert w\vert\vert_2^2$ |
  | Lasso | $\sum_i (y_i - \hat{y_i})^2 + \lambda \vert\vert w\vert\vert_1$ |
  | Elastic Net | $\sum_i (y_i - \hat{y_i})^2 + \lambda_1 \vert\vert w\vert\vert_1+ \lambda_2\vert\vert w\vert\vert_2^2$ |

## ols solution

- $w^*_{OLS} = (X^TX)^{-1}X^Ty$
- 2 derivations: least squares, orthogonal projection

## ridge regression intuition

- $w^*_{RIDGE} = (X^TX \color{red}{+ \lambda I})^{-1}X^Ty$
- ![1d8XV](assets/1d8XV.png)

# parameter estimation

## probabilistic model

- assume a **true underlying model**
- ex. $Y_i \sim \mathcal N(\theta^TX_i, \sigma^2)$
- this is equivalent to $P(Y_i|X_i; \theta) = \mathcal N(\theta^TX_i, \sigma^2)$

## bayes rule

- $\overbrace{p(\theta \vert x)}^{\text{posterior}} = \frac{\overbrace{p(x\vert\theta)}^{\text{likelihood}} \overbrace{p(\theta)}^{\text{prior}}}{p(x)}$
- ![bayes2](assets/bayes2.jpg)

## likelihood 

$\mathcal L = p(data | \theta)$~ product over all n examples
- $p(x|\theta)$?
- $p(y|x; \theta)$?
- $p(x, y | \theta)$?
- $\to$ depends on the problem

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

![bias-and-variance](assets/bias-and-variance.jpg)

## intuition 2

![fittings](assets/fittings.jpg)

## bias

- bias of a method: $E[\hat{f}(x) - f(x)]$
  - expectation over training sets with fixed x
- could also have bias of a point estimate: $E[\hat{\theta} - \theta]$

## variance

- "estimation error"
- variance of a method: $V[\hat{f}(x)] = E[\big(\hat{f}(x) - E[\hat{f}(x)]\big)^2]$
  - expectation over training sets with fixed x

## bias-variance trade-off

- mean-squared error of method: $E[(\hat{f}(x) - f(x))^2]$
  - = bias$^2$ + variance
  - = $E[\hat{f}(x) - f(x)]^2$ + $E[(\hat{f(x)} - E[\hat{f(x)}])^2]$

![biasvariance](assets/biasvariance.png)

# multivariate gaussian

## definitions

- $p(x\vert\mu, \Sigma) = \frac{1}{(2\pi )^{n/2} \vert\Sigma\vert^{1/2}} \exp\left[ -\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right]$
  - $\mu$ is mean vector
  - $\Sigma$ is covariance matrix

![MultivariateNormal](assets/MultivariateNormal.png)

## understanding $\Sigma$

![main-qimg-988175a0da2cb7674391d43a2edab558](assets/main-qimg-988175a0da2cb7674391d43a2edab558.png)

## understanding $\Sigma^{-1}$

![slide_11](assets/slide_11.jpg)

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

![Heteroscedasticity](assets/Heteroscedasticity.jpg)

## overview

![Screen Shot 2018-06-24 at 7.40.57 PM](assets/Screen Shot 2018-06-24 at 7.40.57 PM.png)

## total LS intuition

- add i.i.d. gaussian noise in x and y - regularization

![Screen Shot 2018-06-29 at 4.08.09 PM](assets/Screen Shot 2018-06-29 at 4.08.09 PM.png)

## total LS solution

- $\hat{w}_{TLS} = (X^TX - \sigma^2 I)^{-1}X^Ty$
  - here, $\sigma$ is last singular value of $[X \: y]$

# dimensionality reduction

## pca intuition

orthogonal dimensions that maximize variance of $X$

![pca](assets/pca.png)

## pca in python

```python
X -= np.mean(X, axis = 0) #zero-center data (nxd) 
cov = np.dot(X.T, X) / X.shape[0] #get cov. matrix (dxd) 
U, D, V = np.linalg.svd(cov) #compute svd, (all dxd) 
Xrot_reduced = np.dot(X, U[:, :2]) #project in 2d (nx2)
```

## pca in practice

- eigenvalue represents prop. of explained variance: $\sum \lambda_i = tr(\Sigma) = \sum Var(X_i)$	
- use svd
- adaptive PCA is faster (sequential)

## cca

- linearly independent dimensions that maximize correlation between $X, Y$
- invariant to scalings / affine transformations of X, Y

  ![cca](assets/cca.jpg)

## correlations

![correlations](assets/correlations.png)

# kernels

## why are kernels useful?

![data_2d_to_3d_hyperplane](assets/data_2d_to_3d_hyperplane.png)

## ex. ridge regression

- reformulate the problem to be computationally efficient + nonlinear
  - matrix inversion is ~$O(dim^3)$
- $\hat{w} = (\color{red}{\underbrace{X^TX}_{dxd}} + \lambda I)^{-1}X^Ty$ ~ faster when $\color{red}{d << n}$
- $\hat{w} = X^T(\color{red}{\underbrace{XX^T}_{nxn}} + \lambda I)^{-1}y$ ~ faster when $\color{red}{n << d}$

## kernels

![Screen Shot 2018-06-24 at 9.53.55 PM](assets/Screen Shot 2018-06-24 at 9.53.55 PM.png)

- $\phi_i^T\phi_j = \phi(x_i)^T \phi(x_j)$

## kernel trick ex.

- ex. with degree 2, 2 variables: $\phi(\mathbf x) = \begin{bmatrix} x_1^2 & x_2^2 &\sqrt{2}x_1x_2 & \sqrt{2}x_1 & \sqrt{2}x_2 &1\end{bmatrix}^T $
- $\mathbf{x_i} = [x_1, x_2]$
- $\phi(\mathbf x_i)^T \phi (\mathbf x_j) = (\mathbf x_i^T \mathbf x_j + 1)^2$
  - left is O(augmented feature space)
  - right is O(original feature space + log(polynomial degree))
- also works for other kernels...

## different from kernel regression...

- note, what discussed here is different from the nonparametric technique of kernel regression: 
- ```$\widehat{y}_h(x)=\frac{\sum_{i=1}^n K_h(x-x_i) y_i}{\sum_{j=1}^nK_h(x-x_j)} $```
  - K is a kernel with a bandwidth h

# optimization problems

## overview

- minimizing things
- ex. $\underset{\theta}{arg min} \: \sum \big(y_i - f(x_i; \theta)\big)^2$

![loss_surfaces](assets/loss_surfaces.jpg)

## convexity

- Hessian $\nabla^2 f(x) \succeq 0 \: \forall x$

- $f(x_2) \geq f(x_1) + \nabla f(x_1) (x_2 - x_1)$

  ![ tangents](assets/ tangents.png)

## convexity continued

- $\color{purple}{t f(x_1) + (1-t) f(x_2)} \geq f(tx_1 + (1-t)x_2)$

![1200px-ConvexFunction.svg](assets/1200px-ConvexFunction.svg.png)

## strong convexity + smoothness

$\color{red}0 \preceq \color{green}{\underset{\text{strong convexity}}{mI}} \preceq \nabla^2 \color{cornflowerblue}{f(x)} \preceq \underset{\text{smoothness}}{MI}$

![bounds](assets/bounds.jpg)

## smoothness

- M-smooth = Lipschitz continuous gradient: $||\nabla f(x_2) - \nabla f(x_1)|| \leq M||x_2-x_1||\quad \forall x_1,x_2$

  ![lipschitz](assets/lipschitz.jpg)

# optimization algorithms

## gradient descent

![0_QwE8M4MupSdqA3M4](assets/0_QwE8M4MupSdqA3M4.png)

## when do we stop?

- validation error stops changing
- changes become small enough

## stochastic gradient descent

![stochastic-vs-batch-gradient-descent](assets/stochastic-vs-batch-gradient-descent.png)



## [momentum demo](https://distill.pub/2017/momentum/)

- $$\theta^{(t+1)} = \theta^{(t)} - \alpha_t \nabla f(\theta^{(t)}) + \color{cornflowerblue}{\underset{\text{momentum}}{\beta_t (f(\theta^{(t)}) - f(\theta^{(t-1)}))}}$$

  <div class="divmomentum">
      <iframe class="iframemomentum" src="https://distill.pub/2017/momentum/" scrolling="no" frameborder="no"></iframe>
  </div>

  <style>
  .divmomentum {
      position: relative;
      width: block;
      height: 600px;
      overflow: hidden;
  }

  .iframemomentum {
  ​    position: absolute;            
  ​    top: -165px;
  ​    left: -25px;
  ​    width: 1424px;
  ​    height: 768px;
  }

  </style>

## newton-raphson

![slide_8](assets/slide_8.jpg)
- apply to find roots of **f'(x)**: $\theta^{(t+1)} = \theta^{(t)} - \nabla^2 f(\theta^{(t)})^{-1}\nabla f(\theta^{(t)})$

## gauss-newton

- modify newton's method assuming we are minimizing nonlinear least squares
- $\theta^{(t+1)} = \theta^{(t)} - \nabla^2 f(\theta^{(t)})^{-1}\nabla f(\theta^{(t)})$
- $\theta^{(t+1)} = \theta^{(t)} + \color{cadetblue}{(J^TJ)}^{-1} \color{cadetblue}{J^T\Delta y}$ $\quad J$ is the Jacobian

# neural nets

## so much hype

- it predicts: vision, audio, text, ~rl
- it's easy: feature engineering
- it's magical: huge, but doesn't overfit

## perceptron

![perceptron](assets/perceptron.png) ![perceptron-1](assets/perceptron-1.gif)

## training a perceptron

- loss function: $L(x, y; w) = (\hat{y} - y)^2$
- goal: $\frac{\partial L}{\partial w_i}$ for all weights
- calculate efficiently with backprop

## [backprop demo](https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/)
<div class="divnn1">
    <iframe class="iframenn1" src="https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/" scrolling="yes" frameborder="no"></iframe>
</div>

<style>
.divnn1 {
    position: relative;
    width: block;
    height: 1200px;
    overflow: hidden;
}

.iframenn1 {
​    position: absolute;            
​    top: -80px;
​    left: 2px;
​    height: 1000px;
​    width: 1000px;
​    -webkit-transform: scale(0.75);
}
</style>

## [nn demo playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.63885&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

## going deeper

![1_gccuMDV8fXjcvz1RSk4kgQ](assets/1_gccuMDV8fXjcvz1RSk4kgQ.png)

<div class="divnn">
    <iframe class="iframenn" src="https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.63885&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false" scrolling="no" frameborder="no"></iframe>
</div>

<style>

.divnn {
​    position: relative;
​    width: block;
​    height: 800px;
​    overflow: hidden;
}

.iframenn {
​    position: absolute;            
​    top: -130px;
​    left: 2px;
​    width: 1800px;
​    height: 625px;
}

</style>

## coding DNNs in numpy

```python
from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
for iteration in xrange(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))
```



## coding DNNs in advanced numpy

```python
import tensorflow as tf
import torch
```

## cnns

![cnns](assets/cnns.gif)

## cnns 2

![cnn2](assets/cnn2.jpeg)

## rnns

![RNN-longtermdependencies](assets/RNN-longtermdependencies.png)

## connection to the brain?

![30124068_2022213811328985_1877822313844441088_o](assets/30124068_2022213811328985_1877822313844441088_o.png)

# discriminative vs. generative

## definitions

![discriminative_vs_generative](assets/discriminative_vs_generative.png)

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

![Exam_pass_logistic_curve](assets/Exam_pass_logistic_curve.jpeg)

- $\sigma(z) = \frac{1}{1+e^{-z}}$
- $P(\hat{Y}=1|x; w) = \sigma(w^Tx)$
  - threshold to predict
  - not really regression

## comparison with OLS

![VVtRW](assets/VVtRW.png)

![nEC4H](assets/nEC4H.png)

## loss functions

- log-loss = cross-entropy: $-\sum_x p(x) \: log \: q(x)$
  - $p(x)$ true $y$
  - $q(x)$ predicted probability of y
- corresponds to MLE for Bernoulli

![Screen Shot 2018-07-02 at 11.26.42 AM](assets/Screen Shot 2018-07-02 at 11.26.42 AM.png)

## multiclass

- one-hot encoding: $[1, 0, 0]$, $[0, 1, 0]$, $[0, 0, 1]$
- softmax function: $\sigma(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_j \exp z_j}$
- loss function still cross-entropy

![multihierarchy](assets/multihierarchy.png)

## training

- no closed form, but convex loss $\implies$ convex optimization!
  - minimize loss on cross-entropy (where p(x) is modelled by sigmoid)
  - or maximize likelihood

# gaussian discriminant analysis

## generative model

![comparisons](assets/comparisons.png)

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

![lda_1](assets/lda_1.png)



## multiclass lda vs. qda

![Screen Shot 2018-07-21 at 10.41.16 AM](assets/Screen Shot 2018-07-21 at 10.41.16 AM.png)

# em

## k-means (2d)

![k-means](assets/k-means.gif)

## mixture of gaussians (2d)



![](assets/inside-cluster-em.gif)![](assets/ad8e9b45e3d01deef10f0cc07ec22144c3c631b3.gif)



## mixture of gaussians (1d)

![mixture-iterations](assets/mixture-iterations.gif)

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

![Screen Shot 2018-07-02 at 3.37.07 PM](assets/Screen Shot 2018-07-02 at 3.37.07 PM.png)

- doesn't find best solution
- unstable when data not linearly separable

## what's w?

```$\hat{y} =\begin{cases}   1 &\text{if } w^Tx +b \geq 0 \\ -1 &\text{otherwise}\end{cases}$```

![svm_w](assets/svm_w.png)

## how far are points?

decision boundary: {$x: w^Tx - b = 0$}

$D = \frac{|w^T(z-x_0)|}{||w||_2} = \frac{|w^Tz-b|}{||w||_2}$

![svm_proj](assets/svm_proj.png)

## hard margin intuition

```$\begin{align} \underset{m, w, b}{\max} \quad &m \\ s.t. \quad &y_i \frac{(w^Tx_i-b)}{||w||_2} \geq m \: \forall i\\ &m \geq 0\end{align}$```

## hard margin formulation

- let $m = 1 / ||w||_2 \implies$unique soln

```$\underset{w, b}{\min}\quad \frac{1}{2} ||w||_2^2 \\s.t. \quad y_i (w^Tx_i - b) \geq 1\: \forall i$```

## ${\color{cadetblue}{\text{soft}}}$ margin

```$\begin{align}\underset{w, b, \color{cadetblue}\xi}{\min}\quad &\frac{1}{2} ||w||_2^2 \color{cadetblue}{+C\sum_i \xi_i}\\s.t. \quad &y_i (w^Tx_i + b) \geq 1 \color{cadetblue}{-\xi_i}\: \forall i\\ &\color{cadetblue}{\xi_i \geq 0 \: \forall i}\end{align}$```

![errs](assets/errs.png)

## binary classification

can rewrite by absorbing $\xi$ constraints

$\underset{w, b}{\min} \quad \frac{1}{2}||w||^2 + C\sum_i \max(1-y_i(w^Tx_i - b), 0)$

## summarizing models

![losses](assets/losses.png)

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

![Screen Shot 2018-07-03 at 12.14.53 AM](assets/Screen Shot 2018-07-03 at 12.14.53 AM.png)

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


![Screen Shot 2018-07-03 at 9.16.55 AM](assets/Screen Shot 2018-07-03 at 9.16.55 AM.png)

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

![Screen Shot 2018-07-03 at 9.51.10 AM](assets/Screen Shot 2018-07-03 at 9.51.10 AM.png)

## orthogonal matching pursuit

- at each step, update all nonzero weights

# decision trees / random forests

## decision tree intuition

![decision](assets/decision.png)

## training

- greedy - use metric to pick attribute
  - split on this attribute then repeat
  - high variance

## information gain

maximize H(parent) - [weighted average] $\cdot$ H(children)

- often picks too many attributes

![c50](assets/c50.png)


## info theory

- maximize $I(X; Y) \equiv$ minimize $H(Y|X)$

![entropy-venn-diagram](assets/entropy-venn-diagram.png)

## split functions
- info gain (approximate w/ gini impurity)
- misclassification rate
- (40-40);  could be: (30-10, 10-30), (20-40, 20-0)

![errs-0983654](assets/errs-0983654.png)

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

- train more *weak learners* to fix current errors

## adaboost

- initialize weights to 1/n
- iterate
  - classify weighted points
  - re-weight points based on errors
- finally, output error-weighted sum of weak learners

## adaboost comments

- derived using exponential loss risk minimization
- test error can keep decreasing once training error is at 0

## gradient boosting

- want to subtract gradient of loss with respect to current total model
- models need not be differentiable
- for squared loss, just the residual

<style>

.reveal h1,
.reveal h2,
.reveal h3,
.reveal h4,
.reveal h5,
.reveal h6 {
​	text-transform: lowercase;
}

.reveal section img { 
​	background:none; 
​	border:none; 
​	box-shadow:none; 
​	filter: invert(1); 
}

iframe {
​    filter: invert(1);
}


body {
  background: #000;
  background-color: #000; 
}



</style>