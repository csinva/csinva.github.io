---
layout: notes
section-type: notes
title: qi notes
category: ai
---

* TOC
{:toc}

# linear discriminant analysis
- PCA - find component axes that maximize the variance of our data
	- "unsupervised" - ignores class labels
- LDA - maximize the separation between multiple classes
	- "supervised" - computes directions (linear discriminants) that represent axes that maximize separation between multiple classes
- used as dimensionality reduction technique 
- project a dataset onto a lower-dimensional space with good class-separability

# datasets
- resting state fMRI gives a time-series of things turning on
	- we want to model correlations between everything, we use a gaussian graphical model
- *brain atlas* - serial sections of brain images
- *histology* - the study of the microscopic structure of tissuesch
- leave-one-out cross validation and try to classify autism / non-autism
	- see how well autism subjects are identified
- how good is the final connectome?
1. ABIDE 
	- normal group ~500
		- has brain imaging
		- subjects as rows, features as cols
			- each feature can be an ROI
	- autism group ~500
		- has brain imaging
	- no molecular measurements
2. ABA data
	- has both genotype & phenotype level data
	- don't know what kind of phenotype
	- mostly about genotype data
	- human ROI has ~200
	- in clustering case, 30000 recordings, need to cluster into groups

# algorithms
- SLFA
	- the features (ex. ROI) have clusters
	- SLFA tries to find dependencies between the clusters instead of the variables
	- group and group dependency
	- this works better in genotype case
- SIMULE
	- context-sensitive graph

# kendall tau
- https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
- (# concordant pairs - # discordant pairs) / (n*(n-1)/2)
	- concordant if both ranks agree
	- must do something special if tied
- matlab has pretty fast implementation, R slow
- want to speed up / parallelize this - use multicore, not gpu
- the Kendall correlation is not affected by how far from each other ranks are but only by whether the ranks between observations are equal or not

# gaussian graphical model
- likelihood = probability of seeing data given parameters of the model
- clustering algorithm - associate conditional probability with each node (relation between the nodes, given all the other nodes)
- http://www.cis.upenn.edu/~mkearns/papers/barbados/jordan-tut.pdf
- the weights in a network make local assertions about the relationships between neighboring nodes
	- inference algorithms turn these local assertions into global assertions about the relationships between nodes
- $P(A|B) = P(AB) / P(B)$
- can be used for learning (given inputs, outputs)
- A *Gaussian graphical model* is a graph in which all random variables are continuous and jointly Gaussian.
- see defs.png
- *precision matrix* - inverse of covariance matrix; gives pairwise correlations

# graphical lasso
- optimize parameter to minimize regression between Y and B*X
- problem is hard because far less samples than nodes so can't invert covariance matrix
- coordinate-descent methods: optimize over one variable at a time
- l1-normalization makes it so there have to be a lot of 0s in B
	- so does l0, but this is harder to solve
- have regress all the variables against all the other variables
	- graphical lasso lets us do this very efficiently with coordinate descent

# structure learning
- *structure learning* aims to discover the topology of a probabilistic network of variables such that this network represents accurately a given dataset while maintaining low complexity
- *accuracy* of representation - likelihood that the model explains the observed data
- *complexity* of a graphical model - number of parameters

# representational similarity learning
- aims to discover features that are important in representing (human-judged) similarities among objects
- can be posed as a sparsity-regularized multi-task regression problem
- related to *representational similarity analysis*

# latent dirichlet allocation
- generative model - explain observations from unobserved variables
- In LDA, each document may be viewed as a mixture of various topics
- similar to probabilistic latent semantic analysis (pLSA), except that in LDA the topic distribution is assumed to have a Dirichlet prior

# latent variable model
- relates *manifest variables* to *latent variables*
- responses on the manifest variables are result of latent variables
	- manifest variables have *local independence* - nothing in common after controlling for latent variable
- latent factor models have been proposed to find concise descriptions of data