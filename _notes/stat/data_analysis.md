---
layout: notes
section-type: notes
title: data analysis
category: stat
---
* TOC
{:toc}
# pqrs

- Goal: *inference* - conclusion or opinion formed from evidence
- *PQRS* 
  - P - population
  - Q - question - 2 types
    1. hypothesis driven - does a new drug work
    2. discovery driven - find a drug that works
  - R - representative data colleciton
    - simple random sampling = *SRS*
      - w/ replacement: $var(\bar{X}) = \sigma^2 / n$
      - w/out replacement: $var(\bar{X}) = (1 - \frac{n}{N}) \sigma^2 / n$ 
  - S - scrutinizing answers


# visual summaries

- numerical summaries
  - mean vs. median
  - sd vs. iq range
- visual summaries
  - histogram
  - *kernel density plot* - Gaussian kernels
    - with *bandwidth* h $K_h(t) = 1/h K(t/h)$
- plots
  1. box plot / pie-chart
  2. scatter plot / q-q plot
    - *q-q plot* - easily check normality
      - plot percentiles of a data set against percentiles of a theoretical distr.
      - should be straight line if they match
  3. transformations = feature engineering
    - log/sqrt make long-tail data more centered and more normal
    - **delta-method** - sets comparable bw (wrt variance) after log or sqrt transform: $Var(g(X)) \approx [g'(\mu_X)]^2 Var(X)$ where $\mu_X = E(X)$
  4. *least squares*
    - inversion of pxp matrix ~O(p^3)
    - regression effect - things tend to the mean (ex. bball children are shorter)
    - in high dims, l2 worked best
  5. kernel smoothing + lowess
    - *nadaraya-watson kernel smoother* - locally weighted scatter plot smoothing
      - $$g_h(x) = \frac{\sum K_h(x_i - x) y_i}{\sum K_h (x_i - x)}$$ where h is bandwidth
    - *loess* - multiple predictors / *lowess* - only 1 predictor
      - also called *local polynomial smoother* - locally weighted polynomial
      - take a window (span) around a point and fit weighted least squares line to that point
      - replace the point with the prediction of the windowed line
      - can use local polynomial fits rather than local linear fits
  6. *silhouette plots* - good clusters members are close to each other and far from other clustersf

     1. popular graphic method for K selection
     2. measure of separation between clusters $s(i) = \frac{b(i) - a(i)}{max(a(i), b(i))}$
       1. a(i) - ave dissimilarity of data point i with other points within same cluster
       2. b(i) - lowest average dissimilarity of point i to any other cluster
     3. good values of k maximize the average silhouette score

# 6 - clustering: k-means, spectral
- *spectral* - does dim reduction on eigenvalues (spectrum) of similarity matrix before clustering in few dims
  - uses adjacency matrix
  - basically like PCA then k-means
  - performs better with regularization - add small constant to the adjacency matrix
- *multidimensional scaling* - given a a distance matrix, MDS tries to recover low-dim coordinates s.t. distances are preserved
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

# 12 - kde, em

- canonical bandwidth
- optimal kernel function - *Epanechnikov*
  - R approximates it w/ cosine kernel (smaller tails)
  - also yields an MSE rate
- ex. mixture of 2 gaussians
- ***EM***
  - goal: maximize $L(\theta)$ for data X and parameters $\theta$
    - equivalent: maximize $\Delta(\theta \| \theta_n) \leq L(\theta) - L(\theta_n)$
      - the function $l(\theta \| \theta_n) = L(\theta_n) + \Delta(\theta \| \theta_n)$ is local concave 
        approximator
      - introduce z (probablity of assignment): $P(X\|\theta) = \sum_z P(X\|z, \theta) P(z\|\theta)$
  - 3 steps
    1. initialize $\theta_1$
    2. E-step - calculate $E_{Z\|X, \theta_n} ln P(X, z \| \theta)$
      - basically assigns z var
      - this is the part of $l(\theta \| \theta_n)$ that actually depends on $\theta$
    3. M-step - $\theta_{n+1} = argmax_{\theta} E_{Z\|X, \theta_n} ln P(X, z \| \theta)$
  - guaranteed to converge to local min of likelihood

