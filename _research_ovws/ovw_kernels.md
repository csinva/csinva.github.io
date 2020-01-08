---
layout: notes
title: kernels
category: research
---


[TOC]

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



# kernels in deep learning

- [To understand deep learning we need to understand kernel learning](https://arxiv.org/abs/1802.01396) - overfitted kernel classifiers can still fit the data well
- [neural tangent kernel](https://arxiv.org/abs/1806.07572) (jacot et al. 2018)
  - at initialization, artificial neural networks (ANNs) are equivalent to Gaussian
    processes in the infinite-width limit
    - evolution of an ANN during training can also be described by a kernel (kernel gradient descent)
  - different types of kernels impose different things on a function (e.g. want more / less low frequencies)
    - gradient descent in kernel space can be convex if kernel is PD (even if nonconvex in the parameter space)
- [Scaling description of generalization with number of parameters in deep learning](https://arxiv.org/abs/1901.01608) (geiger et al. 2019)
  - number of params = N
  - above 0 training err, larger number of params reduces variance but doesn't actually help
    - ensembling with smaller N fixes problem
  - the improvement of generalization performance with N in this classification task originates from reduced variance of fN when N gets large, as recently observed for mean-square regression
- [understanding the neural tangent kernel](https://arxiv.org/pdf/1904.11955.pdf) (arora et al. 2019)
  
  - method to compute the kernel quickly on a gpu
- [On the Inductive Bias of Neural Tangent Kernels](https://arxiv.org/abs/1905.12173) (bietti & mairal 2019)
- [Kernel and Deep Regimes in Overparametrized Models](https://arxiv.org/abs/1906.05827) (Woodworth...Srebro 2019)
  
  - transition between *kernel* and *deep regimes*



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