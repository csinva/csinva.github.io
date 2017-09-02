---
layout: notes
section-type: notes
title: linear algebra
category: math
---

* TOC
{:toc}

# SVD + eigenvectors
### strang 5.1 - intro
- elimination changes eigenvalues
- eigenvector application to diff eqs $\frac{du}{dt}=Au$
	- soln is exponential: $u(t) = c_1 e^{\lambda_1 t} x_1 + c_2 e^{\lambda_2 t} x_2$
- *eigenvalue eqn*: $Ax = \lambda x \implies (A-\lambda I)x=0$
	- set $det(A-\lambda I) = 0$ to get *characteristic polynomial*
- eigenvalue properties
	- 0 eigenvalue signals that A is singular
	- eigenvalues are on the main diagonal when the matrix is triangular
	- checks
		1. sum of eigenvalues = trace(A)
		2. prod eigenvalues = det(A)
- *defective matrices* - lack a full set of eigenvalues

### strang 5.2 - diagonalization
- assume A (nxn) has n eigenvectors
	- S := eigenvectors as columns
	- $S^{-1} A S = \Lambda$ where corresponding eigenvalues are on diagonal of $\Lambda$
- if matrix A has no repeated eigenvalues, eigenvectors are independent
- other S matrices won't produce diagonal
- only diagonalizable if n independent eigenvectors
	- not related to invertibility
	- eigenvectors corresponding to different eigenvalues are lin. independent
- there are always n complex eigenvalues
- eigenvalues of $A^2$ are squared, eigenvectors remain same
- eigenvalues of $A^{-1}$ are inverse eigenvalues
- eigenvalue of rotation matrix is $i$
- eigenvalues for $AB$ only multiply when A and B share eigenvectors
	- diagonalizable matrices *share the same eigenvector* matrix S iff $AB = BA$
	
### strang 5.3 - difference eqs and power $A^k$
- compound interest
- solving for fibonacci numbers
- Markov matrices
	- steady-state Ax = x 
	- corresponds to $\lambda = 1$
- stability of $u_{k+1} = A u_k$
	- stable if all eigenvalues satisfy $|\lambda_i|$  <1
	- neutrally stable if some $|\lambda_i|=1$
	- unstable if at least one $|\lambda_i|$ > 1
- Leontief's input-output matrix
- *Perron-Frobenius thm* - if A is a positive matrix (positive values), so is its largest eigenvalue. Every component of the corresponding eigenvector is also positive.

### strang 6.3 - singular value decomposition
- SVD for any m x n matrix: $A=U\Sigma V^T$
	- U (mxm) are eigenvectors of $AA^T$
	- columns of V (nxn) are eigenvectors of $A^TA$
	- r singular values on diagonal of $\Sigma$ (m x n) - square roots of nonzero eigenvalues of both $AA^T$ and $A^TA$
	- like rotating, scaling, and rotating back
- properties
	1. for PD matrices, $\Sigma=\Lambda$, $U\Sigma V^T = Q \Lambda Q^T$
		- for other symmetric matrices, any negative eigenvalues in $\Lambda$ become positive in $\Sigma$
- applications
	- very numerically stable because U and V are orthogonal matrices
	- *condition number* of invertible nxn matrix = $\sigma_{max} / \sigma_{min}$
	- $A=U\Sigma V^T = u_1 \sigma_1 v_1^T + ... + u_r \sigma_r v_r^T$
		- we can throw away columns corresponding to small $\sigma_i$
	- *pseudoinverse* $A^+ = V \Sigma^+ U^T$

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
    	- like it curves up
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