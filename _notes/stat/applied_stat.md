---
layout: notes
section-type: notes
title: Statistics
category: stat
---
* TOC
{:toc}


# evaluation
- stability
	1. computational stability
		- randomness in the algorithm
	2. generalization stability
		- randomness in the data
		- sampling methods
			1. *bootstrap* - take a sample
				- repeatedly sample from observed sample w/ replacement
				- bootstrap samples has same size as observed sample
			2. *subsampling*
				- sample without replacement
			3. Is *jackknife resampling*
				- subsample containing all but one of the points
- cv
	- cv error is not good when n < complexity of predictor
		- because summands are correlated
		- assume data units are exchangeable
		- can sometimes use this to pick k for k-means
	- types of cv
		1. v-fold
		2. LOOCV
		3. random split - shuffle and repeat
		4. *one-way CV* = *prequential analysis* - keep testing on next data point, updating model

# 3 - 1-dim numerical and visual summaries of data in context
- numerical summaries
	- mean vs. median
	- sd vs. iq range
- visual summaries
	- histogram
	- *kernel density plot* - Gaussian kernels
		- with *bandwidth* h $K_h(t) = 1/h K(t/h)$

# 4 - box-plot, qq plot, data transformation, scatter plot, and LS
1. box plot / pie-chart
2. scatter plot / q-q plot
	- *q-q plot* - easily check normality
		- plot percentiles of a data set against percentiles of a theoretical distr.
		- should be straight line if they match
3. transformations = feature engineering
	- log/sqrt make long-tail data more centered and more normal
	- **delta-method** - sets comparable bw (wrt variance) after log or sqrt transform: $Var(g(X)) \approx [g'(\mu_X)]^2 Var(X)$ where $\mu_X = E(X)$
4. *least squares*
	- l2 is easier to optimize
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
	
# 5 - visualization of high-dim data, pca, and clustering
- *pca*
	- built on svd / eigen decomposition of covariance matrix $\Sigma = X^TX$
	- each eigenvalue represents prop. of explained variance
		- $\sum \lambda_i = tr(\Sigma) = \sum Var(X_i)$
	- *screeplot*  - eigenvalues in decreasing order, look for num dims with kink
	- don't automatically center/normalize, especially for positive data
- lab notes: *silhouette plots* - good clusters members are close to each other and far from other clustersf
	- popular graphic method for K selection
	- measure of separation between clusters $s(i) = \frac{b(i) - a(i)}{max(a(i), b(i))}$
		- a(i) - ave dissimilarity of data point i with other points within same cluster
		- b(i) - lowest average dissimilarity of point i to any other cluster
	- good values of k maximize the average silhouette score


# 6 - clustering: k-means, spectral
- *spectral* - does dim reduction on eigenvalues (spectrum) of similarity matrix before clustering in few dims
	- uses adjacency matrix
	- basically like PCA then k-means
	- performs better with regularization - add small constant to the adjacency matrix
- lab notes
	- *multidimensional scaling* - given a a distance matrix, MDS tries to recover low-dim coordinates s.t. distances are preserved
		- minimizes goodness-of-fit measure called *stress* = $\sqrt{\sum (d_{ij} - \hat{d}_{ij})^2 / \sum d_{ij}^2}$
		- visualize in low dims the similarity between individial points in high-dim dataset
		- classical MDS assumes Euclidean distances and uses eigenvalues

# 7 - hierarchical clustering, prediction
- link closest points
- need *linkage function* for a cluster
	1. complete - farthest
	2. average - average
	3. single - closest
- ex. MST - keep linking shortest link
- *ultrametric distance* - tighter than triangle inequality
	- $d(x, y) \leq max(d(x,z), d(y,z))$
- k-means++ - better at not getting stuck in local minima
	- randomly move centers apart
- prediction
	1. *replication*
	2. *extrapolation*

	
# 8/9 - prediction, supervised / unsupervised learning
- feature engineering - domain knowledge essential
- evaluate w/ prediction metric + stability
	- can validate assumptions
	- computational cost
	- interpretability
	- stability
- LS - minimize $\|\|Y-Xw\|\|$
	- $\hat{\theta} = (X^TX)^{-1} X^TY$
		- 1. set deriv and solve
		- 2. use projection matrix H to show HY is proj of Y onto R(X)
			- define projection matrix $H = X(X^TX)^{-1} X^T$
				- show $\|\|Y-X \theta\|\|^2 \geq \|\|Y - HY\|\|^2$
				- key idea: subtract and add HY
			- ***pf in pics***
	- if feature correlated, weights aren't stable / can't be interpreted
	- curvature inverse $(X^TX)^{-1}$ - dictates stability
- LS doesn't work when p >> n because of colinearity of X columns
- lasso can help stability somewhat

# 10 - linear decompositions of data matrix, sources of randomness
- stability principle
	1. perturbations to data
	2. perturbations to models
	3. stability measures
		- ex. similarity metric accross clusters / matrices
- linear decompositions: learn D s.t. $X=DA$
	1. NMF - $min_{D \geq 0, A \geq 0} \|\|X-DA\|\|_F^2$
	2. ICA
		- remove correlations and higher order dependence
		- all components are equally important
	3. PCA - orthogonaltiy
			- compress data, remove correlations
	4. K-means - can be viewd as a linear decomposition

# 11 - stat inference, srs, kde under iid
- *inference* - conclusion or opinion formed from evidence
	- P - population
	- Q - question - 2 types
		1. hypothesis driven - does a new drug work
		2. discovery driven - find a drug that works
	- R - representative data colleciton
		- simple random sampling = *SRS*
			- w/ replacement: $var(\bar{X}) = \sigma^2 / n$
			- w/out replacement: $var(\bar{X}) = (1 - \frac{n}{N}) \sigma^2 / n$ 
	- build model
	- S - scrutinizing answers
- *bias/variance trade-off*
	- defs
		- bias sometimes called approximation err
		- variance called estimation err
	- ex. ***estimator for kde***: $\hat{f_{n, h}(x)} = \frac{1}{n}\sum_i K_h (X_i - x)$
		- smooths voxel-wise output
		- $bias = E[\hat{f}(x)] - f(x) = f''(x)/2 \int t^t K(t) dt \cdot h^2$ + smaller order
		- $variance =Var[\hat{f}(x)] = 1/n^2 \sum Var[Y_i] + \frac{2}{n^2} \sum_{i<j} Cov(Y_i, Y_j)$
	- ex. $mse = E[\hat{f}_h(x) - f(x)]^2 = bias^2 + variance$
		- define $\hat{f}_h(x) = \hat{f}_{n,h}(x)$ - same model for each voxel
		- ***bias and variance for this?***
		- define mean L2 err = *risk* = $\int mse(x) dx$
			- minimizing this yields an asymptotically optimal bandwidth
	
# 12 - kde, em, neyman-rubin model
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

# 14 - linear regression and stat. inference
- real slope - should be for the population
- normal lin reg
	- $y_i ~ N(\theta^Tx, \sigma^2)$
	- max likelihood for ***MSE***
	
# 15 - linear reg. and stat. inference II
- normal linear regression
	- variance MLE $\hat{\sigma}^2 = \sum (y_i - \hat{\theta}^T x_i)^2 / n$
		- in unbiased estimator, we divide by n-p
	- LS has a distr. $N(\theta, \sigma^2(X^TX)^{-1})$
- linear regression model
	- when n is large, LS estimator ahs approx normal distr provided that X^TX /n is approx. PSD
- *confidence interval* - if we remade it 100 times, 95 would contain the true $\theta_1$
- type 1 err - like the tails of null distr.
	- *stat. significant*: p = 0.05
	- *highly stat. significant*: p = 0.01
- *weighted LS*: minimize $\sum [w_i (y_i - x_i^T \theta)]^2$
	- $\hat{\theta} = (X^TWX)^{-1} X^T W Y$
	- heteroscedastic normal lin reg model: erorrs ~ N(0, 1/w_i)
- *leverage scores* - measure how much each $x_i$ influences the LS fit
	- for data point i, $H_{ii}$ is the leverage score
- *LAD (least absolute deviation)* fit
	- MLE estimator when error is Laplacian
- *logistic regression*
	- $p_i = P(Y_i=1\|x_i) = exp(x_i^T \theta) / (1+exp(x_i^T \theta))$
	- $Logit(p_i) = log(p_i / (1-p_i)) = x_i^T \theta$