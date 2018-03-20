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
  - FA (factor analysis) - like PCA but with errors, not biased by variance
  - NMF - $min_{D \geq 0, A \geq 0} \|\|X-DA\|\|_F^2$
    - SEQNMF
  - ICA
    - remove correlations and higher order dependence
    - all components are equally important
  - PCA - orthogonality
    - compress data, remove correlations
  - LDA/QDA - finds basis that separates classes
  - K-means - can be viewed as a linear decomposition
- dynamics
  - LDS/GPFA
  - NLDS
- sparse coding
- *spectral* clustering - does dim reduction on eigenvalues (spectrum) of similarity matrix before clustering in few dims
  - uses adjacency matrix
  - basically like PCA then k-means
  - performs better with regularization - add small constant to the adjacency matrix

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
-  nonlinear pca
    - usually uses an auto-associative neural network
      ​	


# ica

- like PCA, but instead of the dot product between components being 0, the mutual info between components is 0
- goals
  - minimizes statistical dependence between its components
  - maximize information transferred in a network of non-linear units
  - uses information theoretic unsupervised learning rules for neural networks
- problem - doesn't rank features for us


# lda / qda
- reduced to axes which separate classes (perpendicular to the boundaries)


# t-sne / umap
- t-sne preserves pairwise neighbors
- UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction



# multidimensional scaling (MDS)

- given a a distance matrix, MDS tries to recover low-dim coordinates s.t. distances are preserved
- minimizes goodness-of-fit measure called *stress* = $\sqrt{\sum (d_{ij} - \hat{d}_{ij})^2 / \sum d_{ij}^2}$
- visualize in low dims the similarity between individial points in high-dim dataset
- classical MDS assumes Euclidean distances and uses eigenvalues
  - constructing configuration of n points using distances between n objects
  - uses distance matrix
    - $d_{rr} = 0$
    - $d_{rs} \geq 0$
  - solns are invariant to translation, rotation, relfection
  - solutions types
    1. non-metric methods - use rank orders of distances
       - invariant to uniform expansion / contraction
    2. metric methods - use values
  - D is *Euclidean* if there exists points s.t. D gives interpoint Euclidean distances
    - define B = HAH
      - D Euclidean iff B is psd