---
layout: notes
section-type: notes
title: Dimensionality Reduction
category: ai
---
* TOC
{:toc}

# PCA
- have p random variables
- want to develop a new set of K axes (linear combinations of the original p axes) in the direction of greatest variability
    - this is best for visualization, reduction, classification, noise reduction
- to find axis - minimize sum of squares of projections onto line =($v^TX^TXv$ subject to $v^T v=1$ )
    - $\implies v^T(X^TXv-\lambda v)=0$
- eigenvectors of covariance matrix -> principal components
    - most important corresponds to largest eigenvalue (eigenvalue corresponds to variance)
- finding eigenvectors can be hard to solve, so 3 other methods
	1. singular value decomposition (SVD)
	2. multidimensional scaling (MDS)
		- based on eigenvalue decomposition
	3. adaptive PCA
		- extract components sequentially, starting with highest variance so you don't have to extract them all	
		
### nonlinear PCA
- usually uses an auto-associative neural network
		
# ICA
- like PCA, but instead of the dot product between components being 0, the mutual info between components is 0- goals	- minimizes statistical dependence between its components	- maximize information transferred in a network of non-linear units	- uses information theoretic unsupervised learning rules for neural networks- problem - doesn't rank features for us