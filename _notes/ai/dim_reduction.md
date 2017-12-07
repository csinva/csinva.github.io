---
layout: notes
section-type: notes
title: Dimensionality Reduction
category: ai
---
* TOC
{:toc}
# overview

- linear decompositions: learn D s.t. $X=DA$
  - NMF - $min_{D \geq 0, A \geq 0} \|\|X-DA\|\|_F^2$
  - ICA
    - remove correlations and higher order dependence
    - all components are equally important
  - PCA - orthogonaltiy
    - compress data, remove correlations
  - K-means - can be viewed as a linear decomposition
  - sparse coding???

# pca

- have p random variables
- want new set of K axes (linear combinations of the original p axes) in the direction of greatest variability
    - this is best for visualization, reduction, classification, noise reduction
- to find axis - minimize sum of squares of projections onto line =($v^TX^TXv$ subject to $v^T v=1$ )
    - $\implies v^T(X^TXv-\lambda v)=0$
- SVD: let $X = U D V^T$
  - $V_q$ (pxq) is first q columns of V
    - $H = V_q V_q^T$ is the *projection matrix*
    - to transform $x = Hx$
  - columns of $UD$ (Nxp) are called the *principal components* of X
- eigenvectors of covariance matrix -> principal components
    - most important corresponds to largest eigenvalue (eigenvalue corresponds to variance)
- finding eigenvectors can be hard to solve, so 3 other methods
  1. singular value decomposition (SVD)
  2. multidimensional scaling (MDS)
    - based on eigenvalue decomposition
  3. adaptive PCA
    - extract components sequentially, starting with highest variance so you don't have to extract them all	
- good PCA code: http://cs231n.github.io/neural-networks-2/
- *pca*
    - built on svd / eigen decomposition of covariance matrix $\Sigma = X^TX$
    - each eigenvalue represents prop. of explained variance
      - $\sum \lambda_i = tr(\Sigma) = \sum Var(X_i)$
    - *screeplot*  - eigenvalues in decreasing order, look for num dims with kink
      - don't automatically center/normalize, especially for positive data
### nonlinear PCA
- usually uses an auto-associative neural network
  ​	
# ICA
- like PCA, but instead of the dot product between components being 0, the mutual info between components is 0
- goals
  - minimizes statistical dependence between its components
  - maximize information transferred in a network of non-linear units
  - uses information theoretic unsupervised learning rules for neural networks
- problem - doesn't rank features for us