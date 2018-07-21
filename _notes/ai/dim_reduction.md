---
layout: notes
section-type: notes
title: dim. Reduction
category: ai
---
* TOC
{:toc}

---

# overview

| Method              | Analysis objective | Temporal smoothing | Explicit noise model |
|---------------------|--------------------|--------------------|----------------------|
| PCA                 | Covariance         | No                 | No                   |
| FA                  | Covariance         | No                 | Yes                  |
| LDS/GPFA            | Dynamics           | Yes                | Yes                  |
| NLDS                | Dynamics           | Yes                | Yes                  |
| LDA                 | Classification     | No                 | No                   |
| Demixed             | Regression         | No                 | Yes/No               |
| Isomap/LLE          | Manifold discovery | No                 | No                   |
| T-SNE               | ....               | ....               | ...                  |
| UMAP                | ...                | ...                | ...                  |
| NMF                 | ...                | ...                | ...                  |
| SVCCA?              |                    |                    |                      |
| diffusion embedding |                    |                    |                      |
| ICA                 |                    |                    |                      |
| K-means             |                    |                    |                      |
| Autoencoders        |                    |                    |


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

- want new set of axes (linearly combine original axes) in the direction of greatest variability
    - this is best for visualization, reduction, classification, noise reduction
    - assume $X$ (nxp) has zero mean

- derivation: 

    - minimize variance of X projection onto a unit vector v: ($\frac{1}{n} \sum (x_i^Tv)^2 = \frac{1}{n}v^TX^TXv$ subject to $v^T v=1$ )
    - $\implies v^T(X^TXv-\lambda v)=0​$... solution is achieved when $v​$ is eigenvector corresponding to largest eigenvalue
    - like minimizing perpendicular distance between data points and subspace onto which we project

- SVD: let $U D V^T = SVD(Cov(X))$

    - $Cov(X) = \frac{1}{n}X^TX$, where X has been demeaned

- equivalently, eigenvalue decomposition of covariance matrix $\Sigma = X^TX$
  - each eigenvalue represents prop. of explained variance: $\sum \lambda_i = tr(\Sigma) = \sum Var(X_i)$

  - *screeplot*  - eigenvalues in decreasing order, look for num dims with kink
    - don't automatically center/normalize, especially for positive data

- SVD is easier to solve than eigenvalue decomposition, can also solve other ways
  1. multidimensional scaling (MDS)
    - based on eigenvalue decomposition
  2. adaptive PCA
    - extract components sequentially, starting with highest variance so you don't have to extract them all	

- good PCA code: http://cs231n.github.io/neural-networks-2/
```python
X -= np.mean(X, axis = 0) # zero-center data (nxd)
cov = np.dot(X.T, X) / X.shape[0] # get cov. matrix (dxd)
U, D, V = np.linalg.svd(cov) # compute svd, (all dxd)
Xrot_reduced = np.dot(X, U[:, :2]) # project onto first 2 dimensions (n x 2)
```
- nonlinear pca
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

# nonlinear

# t-sne / umap

- t-sne preserves pairwise neighbors
- UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction

# diffusion embedding

