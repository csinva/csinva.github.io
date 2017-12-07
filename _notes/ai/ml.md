---
layout: notes
section-type: notes
title: Machine Learning
category: ai
---
* TOC
{:toc}
# Overview

- 3 types
  - supervised 
  - unsupervised
  - reinforcement
- *bias/variance trade-off*
  - pf
    - ![mse](assets/ml/mse.png)
  - defs
    - bias sometimes called approximation err
    - variance called estimation err
  - ex. ***estimator for kde***: $\hat{f_{n, h}(x)} = \frac{1}{n}\sum_i K_h (X_i - x)$
    - smooths voxel-wise output
    - $bias = E[\hat{f}(x)] - f(x) = f''(x)/2 \int t^t K(t) dt \cdot h^2$ + smaller order
    - $variance =Var[\hat{f}(x)] = 1/n^2 \sum Var[Y_i] + \frac{2}{n^2} \sum_{i<j} Cov(Y_i, Y_j)$
  - ex. $mse = E[\hat{f}_h(x) - f(x)]^2 = bias^2 + variance$
    - define *risk* = mean L2 err = $\int mse(x) dx$
      - minimizing this yields an asymptotically optimal bandwidth

# Feature Selection

## Filtering

- ranks features or feature subsets independently of the predictor
- univariate methods (consider one variable at a time)
  - ex. T-test of y for each variable
  - ex. Pearson correlation coefficient - this can only capture linear dependencies
  - mutual information - covers all dependencies
- multivariate methods
  - features subset selection
  - need a scoring function
  - need a strategy to search the space
  - sometimes used as preprocessing for other methods

## Wrapper

- uses a predictor to assess features of feature subsets
- learner is considered a black-box - use train, validate, test set
- forward selection - start with nothing and keep adding
- backward elimination - start with all and keep removing
- others: Beam search - keep k best path at teach step, GSFS, PTA(l,r), floating search - SFS then SBS

## Embedding
- uses a predictor to build a model with a subset of features that are internally selected
- ex. lasso, ridge regression

# Unsupervised Learning
- labels are not given
- intra-cluster distances are minimized, inter-cluster distances are maximized
- Distance measures
  - symmetric D(A,B)=D(B,A)
  - self-similarity D(A,A)=0
  - positivity separation D(A,B)=0 iff A=B
  - triangular inequality D(A,B) <= D(A,C)+D(B,C)
  - ex. Minkowski Metrics $d(x,y)=\sqrt[r]{\sum \vert x_i-y_i\vert ^r}$
    - r=1 Manhattan distance
    - r=1 when y is binary -> Hamming distance
    - r=2 Euclidean
    - r=$\infty$ "sup" distance
- correlation coefficient - unit independent
- edit distance

## Hierarchical
- Two approaches:
    1. Bottom-up agglomerative clustering - starts with each object in separate cluster then joins
    2. Top-down divisive - starts with 1 cluster then separates
- ex. starting with each item in its own cluster, find best pair to merge into a new cluster
- repeatedly do this to make a tree (dendrogram)
- distances between clustersdefined by *linkage function*
  - single-link - closest members (long, skinny clusters)
  - complete-link - furthest members  (tight clusters)
  - average - most widely used
- ex. MST - keep linking shortest link
- *ultrametric distance* - tighter than triangle inequality
    - $d(x, y) \leq max(d(x,z), d(y,z))$
- k-means++ - better at not getting stuck in local minima
    - randomly move centers apart
- Complexity: $O(n^2p)$ for first iteration and then can only get worse

## Partitional

- partition n objects into a set of K clusters (must be specified)
- globally optimal: exhaustively enumerate all partitions
- minimize sum of squared distances from cluster centroid
- Evaluation w/ labels - purity - ratio between dominant class in cluster and size of cluster

### Expectation Maximization (EM)
- general procedure that includes K-means
- E-step
  - calculate how strongly to which mode each data point “belongs” (maximize likelihood)
- M-step - calculate what each mode’s mean and covariance should be given the various responsibilities (maximization step)
- known to converge
- can be suboptimal
- monotonically decreases goodness measure
- can also partition around medoids
- mixture-based clustering 
- *K-Means*
  - start with random centers
  - assign everything to nearest center: O(\vert clusters\vert *np) 
  - recompute centers O(np) and repeat until nothing changes
  - partition amounts to Voronoi diagram

### Gaussian Mixture Model (GMM)
- continue deriving new mean and variance at each step
- "soft" version of K-means - update means as weighted sums of data instead of just normal mean

# Derivations
## normal equation
- $L(\theta) = \frac{1}{2} \sum_{i=1}^n (\hat{y}_i-y_i)^2$
- $L(\theta) = 1/2 (X \theta - y)^T (X \theta -y)$
- $L(\theta) = 1/2 (\theta^T X^T - y^T) (X \theta -y)$ 
- $L(\theta) = 1/2 (\theta^T X^T X \theta - 2 \theta^T X^T y +y^T y)$ 
- $0=\frac{\partial L}{\partial \theta} = 2X^TX\theta - 2X^T y$
- $\theta = (X^TX)^{-1} X^Ty$

## ridge regression
- $L(\theta) = \sum_{i=1}^n (\hat{y}_i-y_i)^2+ \lambda \vert \vert \theta\vert \vert _2^2$ 
- $L(\theta) = (X \theta - y)^T (X \theta -y)+ \lambda \theta^T \theta$
- $L(\theta) = \theta^T X^T X \theta - 2 \theta^T X^T y +y^T y +  \lambda \theta^T \theta$ 
- $0=\frac{\partial L}{\partial \theta} = 2X^TX\theta - 2X^T y+2\lambda \theta$
- $\theta = (X^TX+\lambda I)^{-1} X^T y$

## single Bernoulli
- L(p) = P(Train | Bernoulli(p)) = $P(X_1,...,X_n\vert p)=\prod_i P(X_i\vert p)=\prod_i p^{X_i} (1-p)^{1-X_i}$
- $=p^x (1-p)^{n-x}$ where x = $\sum x_i$
- $log(L(p)) = log(p^x (1-p)^{n-x}=x log(p) + (n-x) log(1-p)$
- $0=\frac{dL(p)}{dp} = \frac{x}{p} - \frac{n-x}{1-p} = \frac{x-xp - np+xp}{p(1-p)}=x-np$
- $\implies \hat{p} = \frac{x}{n}$

## multinomial
- $L(\theta)=P(Train\vert Multinomial(\theta))=P(d_1,...,d_n\vert \theta_1,...,\theta_p)$ where d is a document of counts x
- =$\prod_i^n P(d_i\vert \theta_1,...\theta_p)=\prod_i^n factorials \cdot \theta_1^{x_1},...,\theta_p^{x_p}$- ignore factorials because they are always same
  \begin{itemize}
- require $\sum \theta_i = 1$
  \end{itemize}
- $\implies \theta_i = \frac{\sum_{j=1}^n x_{ij}}{N}$ where N is total number of words in all docs