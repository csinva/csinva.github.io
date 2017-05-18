---
layout: notes
section-type: notes
title: linear algebra
category: math
---

* TOC
{:toc}

## Linear Basics
- Linear 
    1. Superposition f(x+y) =  f(x)+f(y) 
    2. Proportionality $f(k*x) = k*f(x)$
- Vector Space
    1. Closed under addition
    2. Contains Identity
- Inner Product - returns a scalar
    1. Linear
    2. Symmetric
    3. Something Tricky
- Determinant - sum of products including one element from each row / column with correct sign
- Eigenvalues: $det(A-\lambda I)=0$
- Eigenvectors: Find null space of A-$\lambda$I
- Linearly Independent: $c_1x_1+c_2x_2=0 \implies c_1=c_2=0$
- Mapping $f: a \mapsto b$
    - Onto (surjective): $ \forall b\in B \exists a\in A \, f(a)=b$
    - 1-1 (injective): $f(a_1)=f(a_2) \implies a_1=a_2 $
- norms  $||x||_p = (\sum_{i=1}^n |x_i|^p)^{1/p}$
	- $L_0$ norm - number of nonzero elements
	- $||x||_1 = \sum |x_i|$
	- $||x||_\infty = max_i |x_i|$
- matrix norm - given a vector norm ||X||, matrix norm is given by: ||A|| = $max_{x ≠ 0} ||Ax|| / ||x||$
	- represents the maximum stretching that A does to a vector x
- psuedo-inverse $A^+ = (A^T A)^{-1} A^T$
- inverse
    - if orthogonal, inverse is transpose
    - if diagonal, inverse is invert all elements
    - inverting 3x3 - transpose, find all mini dets, multiply by signs, divide by det
- to find eigenvectors, values
    - $det(A-\lambda I)=0$ and solve for lambdas
    - $A = QDQ^T$ where Q columns are eigenvectors
   
# Singularity
- *nonsingular* = invertible = nonzero determinant = null space of zero
    - only square matrices
    - *rank* of mxn matrix- max number of linearly independent columns / rows
    	- rank==m==n, then nonsingular
    - *ill-conditioned matrix* - matrix is close to being singular - very small determinant
- *positive semi-definite* -  $A \in R^{nxn}$
    - intuitively is like having upwards curve
    - if $\forall x \in R^n, x^TAx \geq 0$ then A is positive semi definite (PSD)
    - if $\forall x \in R^n, x^TAx > 0$ then A is positive definite (PD)
    - PD $\to$ full rank, invertible
- *Gram matrix* - G = $X^T X \implies $PSD
    - if X full rank, then G is PD
    
# Matrix Calculus
- gradient $\Delta_A f(\mathbf{A})$- partial derivatives with respect to each element of matrix
    - f has to be a function that takes matrix, returns a scalar
    - output will be the same sizes as the variable you take the gradient of (in this case A)
    - $\nabla^2$ is not gradient of the gradient
- examples
    - $\nabla_x a^T x = a$
    - $\nabla_x x^TAx = 2Ax$ (if A symmetric)
    - $\nabla_x^2 x^TAx = 2A$ (if A symmetric)
- function f(x,y)
    - 1st derivative is vector of derivatives
    - 2nd derivative is Hessian matrix