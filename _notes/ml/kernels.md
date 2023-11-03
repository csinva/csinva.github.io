---
layout: notes
title: kernels
category: ml
subtitle: An introduction to kernels and recent research.
---

{:toc}

# kernel basics

- basic definition
  - continuous
  - symmetric
  - PSD Gram matrix ($K_n = XX^T$)
- [list of kernels](http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#spline)
- [kernels wiki](https://en.wikipedia.org/wiki/Kernel_method#cite_note-4): kernel memorizes points then uses dists between points to classify
- [learning deep kernels](https://arxiv.org/pdf/1811.08357v1.pdf)
- [learning data-adaptive kernels](https://arxiv.org/abs/1901.07114)
- [kernels that mimic dl](https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf)
- [kernel methods](http://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdfs)
- [wavelet support vector machines](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.412.362&rep=rep1&type=pdf) - kernels using wavelets

  

## ch 4 from [support vector machines book](https://link.springer.com/book/10.1007/978-0-387-77242-4)

- 4.1 - what is a valid kernel
  - in general, most dot-product like things constitute valid kernels
  - a function is a kernel iff it is a symmetric, positive definite function
    - this refers to the $n$ x $n$ matrix with entries $f(x_{row}-x_{col})$ being a psd matrix
  - a given kernel can have many feature spaces (can construct different feature spaces that yield the same inner products)
- 4.2 **reproducing kernel hilbert space (RKHS)** of a kernel
  - hilbert space - abstract vector space with (1) an inner product and (2) is complete (i.e. enough limits so calculus works)
  - the RKHS is the smallest feature space of a kernel, and can serve as a canonical feature space
    - RKHS - a $\mathbb K$-hilbert space that consists of *functions* mapping from X to $\mathbb K$
    - every RKHS has a unique reproducing kernel
    - every kernel has a unique RKHS
  - sums/products of kernels also work


# kernel papers

- [data spectroscopy paper](https://arxiv.org/pdf/0807.3719) (shi et al. 2009)
  - kernel matrix $K_n$
  - Laplacian matrix $L_n = D_n - K_n$
    - $D_n$ is diagonal matrix, with entries = column sums
  - block-diagonal kernel matrix would imply a cluster
    - eigenvalues of these kernel matrices can identify clusters (and are invariant to permutation)
    - want to look for data points corresponding to same/similar eigenvectors
  - hard to know what kernel to use, how many eigenvectors / groups look at
  - here, look at population point of view - realted dependence of spectrum of $K_n$ on the data density function: $K_Pf(x) = \int K(x, y) f(y) dP(y)$

# spectral clustering

- interested in top eigenvectors of $K_n$ and bottom eigenvectors of $L_n$
- scott and longuet-higgins - embed data in space of top eigenvectors, normalize in that space, and investigate block structure
- perona and freeman - 2 clusters by thresholding top eigenvector
- shi & malik - normalized cut: threshold second smallest generalize eigenvector of $L_n$
- similarly we have kernel PCA, spectral dimensionality reduction, and SVMs (which can be viewed as fitting a linear classifier in the eigenspace of $K_n$)