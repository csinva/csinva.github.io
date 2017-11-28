---
layout: notes
section-type: notes
title: Graphical Models
category: cs
---
* TOC
{:toc}

# big data
- marginal correlation - covariance matrix
    - estimates are bad when not n >> d
    - eigenvalues are not well-approximated
    - often enforce sparsity
    - ex. threshold each value in the cov matrix (set to 0 unless greater than thresh) - this threshold can depend on different things
    - can also use regularization to enforce sparsity
    - POET doesn't assume sparsity
- conditional correlation - inverse covariance matrix = precision matrix

# 1 - bayesian networks
- A and B have conditional independence given C if A|B and A|C are independent
	- $$P(AB|C) = P(A|C) P(B|C)$$

### bayesian networks intro
- represented by directed acyclic graph
1. could get an expert to design Bayesian network
2. otherwise, have to learn it from data
- each node is random variable
- weights as tables of conditional probabilities for all possibilities
1. encodes conditional independence relationships
2. compact representation of joint prob. distr. over the variables
- *markov condition* - given its parents, a node is conditionally independent of its non-descendants
- therefore joint distr: $P(X_1 = x_1,...X_n=x_n)=\prod_{i=1}^n P(X_i = x_i | Parents(X_i))$
- *inference* - using a Bayesian network to compute probabilities
	- sometimes have unobserved variables

### sampling
- *exact inference* is feasible in small networks, but takes a long time in large networks
	- approximate inference techniques
1. learning
	- *prior sampling*
		- draw N samples from a distribution S
		- approximate posterior probability based on observed values
	- ex. flip a weighted coin to find out what the probabilities are
		- then move to child nodes and repeat
2. inference
	- suppose we want to know P(D|!A)
		- sample network N times, report probability of D being true when A is false
		- more samples is better
	- *rejection sampling* - if want to know P(D|!A)
		- sample N times, throw out samples where A isn't false
		- return probability of D being true
		- this is slow
	- *likelihood weighting* - fix our evidence variables to their observed values, then simulate the network
		- can't just fix variables - distr. might be inconsistent
		- instead we weight by probability of evidence given parents, then add to final prob
		- for each observation
			- if correct, Count = Count+(1*W)
			- always, Total = Total+(1*W)
		- return Count/Total
		- this way we don't have to throw out wrong samples
		- doesn't solve all problems - evidence only influences the choice of downstream variables
		
### 16 learning overview
- notation
	- *P* - true distribution
	- $\hat{P}$ - sample distribution of P
	- $\tilde{P}$ - estimated distribution P
1. *density estimation* - construct a model $\tilde{M}$ such that $\tilde{P}$ is close to generating $P^*$
	- this can be estimated with *relative entropy distance* = $\mathbf{E_X}\left[ log(\frac{P^*(X)}{\tilde{P(X)}} \right]$
		- also $= - \mathbf{H}_{P^*}(X) - \mathbf{E}_X [log \tilde{P}(X)]$
		- intuitively measures extent of *compression loss (in bits)*
		- can ignore first term because it is unaffected by the model
		- concentrate on *expected log-likelihood* = $\mathcal{l}(D|M) =  \mathbf{E}_X [log \tilde{P}(X)]$
		- maximizes probability of data given the model
		- maximizes prediction assuming we are given complete instances
		- could design test suite of queries to evaluate performance on a range of queries
2. *classification*
	- can set loss function to *classification error* (0/1 loss)
		- this doesn't work well for multiclass labeling
	- *Hamming loss* - counts number of variables Y in which pred differs from ground truth
	- *conditional log-likelihood* = $\mathbf{E}_{x,y ~ P}[log \tilde{P}(y|x)]$ - only measure likelihood with respect to predicted y
3. *knowledge discovery*
	- far more critical to assess the confidence in a prediction
- the amount of data required to estimate parameters reliably grows linearly with the number of parameters, so that the amount of data required can grow exponentially with the network connectivity
- *goodness of fit* - how well does the learned distribution represent the real distribution?

### 17 parameter estimation
- assume *parametric model* P ($x$|$\theta$)
- a *sufficent statistic* can be used to calculate likelihood

### 18 structure learning
- *structure learning* - learning the structure (e.g. connections) in the model

### 20 learning undirected models
- a *potential function* is a non-negative function
- values with higher potential are more probable
- can maximize *entropy* in order to impose as little structure as possible while satisfying constraints