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