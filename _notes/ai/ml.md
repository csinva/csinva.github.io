---
layout: notes
section-type: notes
title: Machine Learning
category: ai
---
* TOC
{:toc}

[toc]

# Overview
- 3 types
	- supervised
	- unsupervised
	- reinforcement

## Evaluation
- accuracy = number of correct classifications / total number of test cases
- balanced accuracy = 1/2 (TP/P + TN/N)
- recall - TP/(TP+FN)
- precision - TP/(TP+FP)
- you train by minimizing SSE on training data
    - report MSE for test samples
    - cross validation - don't have enough data for a test set
    - k-fold - split data into N pieces, test on only 1
    - LOOCV - train on all but one

## Error
- Define a loss function $\mathcal{L}$
\begin{itemize}
- 0-1 loss: |C-f(X)|
- $L_2$ loss: $(C-f(X))^2$
\end{itemize}
- Expected Prediction Error EPE(f) = $E_{X,C} [\mathcal{L}(C,f(X))]$
\begin{itemize}
- =$E_{X}\left[ \sum_i \mathcal{L}(C_i,f(X)) Pr(C_i|X) \right]$
\end{itemize}
- Minimize EPE
\begin{itemize}
- Bayes Classifier minimizes 0-1 loss: $\hat{f}(X)=C_i$ if $P(C_i|X)=max_f P(f|X)$
- KNN minimizes L2 loss: $\hat{f}(X)=E(Y|X)$ 
\end{itemize}
- EPE(f(X)) = $noise^2+bias^2+variance$
\begin{itemize}
- noise - unavoidable
- bias=$E[(\bar{\theta}-\theta_{true})^2]$ - error due to incorrect assumptions
- variance=$E[(\bar{\theta}-\theta_{estimated})^2]$ - error due to variance of training samples
\end{itemize}
- more complex models (more nonzero parameters) have lower bias, higher variance
\begin{itemize}
- if high bias, train and test error will be very close (model isn't complex enough)
\end{itemize}

# Classification
- asymptotic classifier - assume you get infinite training / testing points
\end{itemize}
\begin{itemize}
- discriminative - model P(C|X) directly
\begin{itemize}
- smaller asymptotic error
- slow convergence ~ O(p)
\end{itemize}
- generative - model P(X|C) directly
\begin{itemize}
- generally has higher bias -> can handle missing data
- fast convergence ~ O(log(p))
\end{itemize}
 
## Discriminative
### SVMs
- svm benefits
	1. *maximum margin separator* generalizes well
	2. *kernel trick* makes it very nonlinear
	3. nonparametric - can retain training examples, although often get rid of many
- notation
	- $y \in \{-1,1\}$
	- $h(x) = g(w^tx +b)$
		- g(z) = 1 if $z \geq 0$ and -1 otherwise
- define *functional margin* $\gamma^{(i)} = y^{(i)} (w^t x +b)$
	- want to limit the size of (w,b) so we can't arbitrarily increase functional margin
	- function margin $\hat{\gamma}$ is smallest functional margin in a training set
- *geometric margin* = functional margin / ||w||
	- if ||w||=1, then same as functional margin
	- invariant to scaling of w
- optimal margin classifier
	- want $$max \: \gamma \: s.t. \: y^{(i)} (w^T x^{(i)} + b) \geq \gamma, i=1,..,m; ||w||=1$$
		- difficult to solve, especially because of ||w||=1 constraint
		- assume $\hat{\gamma}=1$ - just a scaling factor
		- now we are maximizing $1/||w||$
	- equivalent to this formulation: $$min \: 1/2 ||w||^2 \: s.t. \: y^{(i)}(w^Tx^{(i)}+b)\geq1, i = 1,...,m$$
- Lagrange duality
- dual representation is found by solving $\underset{a}{argmax} \sum_j \alpha_j - 1/2 \sum_{j,k} \alpha_j \alpha_k y_j y_k (x_j \cdot x_k)$ subject to $\alpha_j \geq 0$ and $\sum_j \alpha_j y_j = 0$
	- convex
	- data only enter in form of dot products, even when predicting $h(x) = sgn(\sum_j \alpha_j y_j (x \cdot x_j) - b)$
	- weights $\alpha_j$ are zero except for *support vectors*
- replace dot product $x_j \cdot x_k$ with *kernel function* $K(x_j, x_k)$
	- faster than just transforming x
	- allows to find optimal linear separators efficiently
- *soft margin* classifier - lets examples fall on wrong side of decision boundary
	- assigns them penalty proportional to distance required to move them back to correct side
- want to maximize margin $M = \frac{2}{\sqrt{w^T w}}$
\begin{itemize}
- we get this from $M=|x^+ - x^-| = |\lambda w| = \lambda \sqrt{w^Tw} $
\end{itemize}
- separable case: argmin($w^Tw$) subject to 
\begin{itemize}
- $w^Tx+b\geq 1$ for all x in +1 class
- $w^Tx+b\leq 1$ for all x in -1 class
\end{itemize}
- solve with quadratic programming
- non-separable case: argmin($w^T w/2 + C \sum_i^n \epsilon_i$) subject to
\begin{itemize}
- $w^Tx_i +b \geq 1-\epsilon_i $ for all x in +1 class
- $w^Tx_i +b \leq -1+\epsilon_i $ for all x in -1 class
- $\forall i, \epsilon_i \geq 0$
- large C can lead to overfitting
\end{itemize}
- benefits
\begin{itemize}
- number of parameters remains the same (and most are set to 0)
- we only care about support vectors
- maximizing margin is like regularization: reduces overfitting
\end{itemize}
- these can be solved with quadratic programming QP
- solve a dual formulation (Lagrangian) instead of QPs directly so we can use kernel trick
\begin{itemize}
- primal: $min_w max_\alpha L(w,\alpha)$
- dual: $max_\alpha min_w L(w,\alpha)$
\end{itemize}
- KKT condition for strong duality
\begin{itemize}
- complementary slackness: $\lambda_i f_i(x) = 0, i=1,...,m$
\end{itemize}
- VC (Vapnic-Chervonenkis) dimension - if data is mapped into sufficiently high dimension, then samples will be linearly separable (N points, N-1 dims)

### kernel functions - new ways to compute dot product (similarity function)
\begin{itemize}
- original testing function: $\hat{y}=sign(\Sigma_{i\in train} \alpha_i y_i x_i^Tx_{test}+b)$
- with kernel function: $\hat{y}=sign(\Sigma_{i\in train} \alpha_i y_i K(x_i,x_{test})+b)$
- linear $K(x,z) = x^Tz$
- polynomial $K (x, z) = (1+x^Tz)^d$
- radial basis kernel $K (x, z) = exp(-r||x-z||^2)$
- computing these is O($m^2$), but dot-product is just O(m)
- function that corresponds to an inner product in some expanded feature space
\end{itemize}
- practical guide
\begin{itemize}
- use m numbers to represent categorical features
- scale before applying
- fill in missing values
- start with RBF
\end{itemize}

### Logistic Regression
- $p = P(Y=1|X)=\frac{exp(\theta^T x)}{1+exp(\theta ^Tx)}$ 
- logit (log-odds) of $p:ln\left[ \frac{p}{1-p} \right] = \theta^T x$
- predict using Bernoulli distribution with this parameter p
- can be extended to multiple classes - multinomial distribution

## Generative
### Naive Bayes Classifier
- let $C_1,...,C_L$ be the classes of Y
- want Posterior $P(C|X) = \frac{P(X|C)(P(C)}{P(X)}$ 
- MAP rule - maximum A Posterior rule
\begin{itemize}
- use Prior P(C)
- using x, predict $C^*=\text{argmax}_C P(C|X_1,...,X_p)=\text{argmax}_C P(X_1,...,X_p|C) P(C)$ - generally ignore denominator
\end{itemize}
- naive assumption - assume that all input attributes are conditionally independent given C
\begin{itemize}
- $P(X_1,...,X_p|C) = P(X_1|C)\cdot...\cdot P(X_p|C) = \prod_i P(X_i|C)$ 
\end{itemize}
- learning
\begin{enumerate}
- learn L distributions $P(C_1),P(C_2),...,P(C_L)$
- learn $P(X_j=x_{jk}|C_i)$ 
\begin{itemize}
- for j in 1:p
- i in 1:$|C|$
- k in 1:$|X_j|$
- for discrete case we store $P(X_j|c_i)$, otherwise we assume a prob. distr. form
\end{itemize}
\begin{itemize}
- naive: $|C| \cdot (|X_1| + |X_2| + ... + |X_p|)$ distributions
- otherwise: $|C|\cdot (|X_1| \cdot |X_2| \cdot ... \cdot |X_p|)$
\end{itemize}
\end{enumerate}
- testing
\begin{itemize}
- P(X|c) - look up for each feature $X_i|C$ and try to maximize
\end{itemize}
- smoothing - used to fill in 0s
\begin{itemize}
- $P(x_i|c_j) = \frac{N(x_i, c_j) +1}{N(c_j)+|X_i|}$ 
- then, $\sum_i P(x_i|c_j) = 1$
\end{itemize}

### Gaussian classifiers
- distributions
\begin{itemize}
- Normal $P(X_j|C_i) = \frac{1}{\sigma_{ij} \sqrt{2\pi}} exp\left( -\frac{(X_j-\mu_{ij})^2}{2\sigma_{ij}^2}\right)$- requires storing $|C|\cdot p$ distributions
- Multivariate Normal $\frac{1}{(2\pi)^{D/2}} \frac{1}{|\Sigma|^{1/2}} exp\left(-\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu)\right)$where $\Sigma$ is covariance matrix
\end{itemize}
\end{itemize}
\begin{itemize}
- decision boundary are points satisfying $P(C_i|X) = P(C_j|X)$
- LDA - linear discriminant analysis - assume covariance matrix is the same across classes
\begin{itemize}
- Gaussian distributions are shifted versions of each other
- decision boundary is linear
\end{itemize}
- QDA - different covariance matrices
\begin{itemize}
- estimate the covariance matrix separately for each class C
- decision boundaries are quadratic
- fits data better but has more parameters to estimate
\end{itemize}
- Regularized discriminant analysis - shrink the separate covariance matrices towards a common matrix
\begin{itemize}
- $\Sigma_k = \alpha \Sigma_k + (1-\alpha) \Sigma$
\end{itemize}
- treat each feature attribute and class label as random variables
\begin{itemize}
- we assume distributions for these
- for 1D Gaussian, just set mean and var to sample mean and sample var
\end{itemize}

### Text classification
- bag of words - represent text as a vector of word frequencies X
\begin{itemize}
- remove stopwords, stemming, collapsing multiple - NLTK package in python
- assumes word order isn't important
- can store n-grams
\end{itemize}
- multivariate Bernoulli: $P(X|C)=P(w_1=true,w_2=false,...|C)$
- multivariate Binomial: $P(X|C)=P(w_1=n_1,w_2=n_2,...|C)$
\begin{itemize}
- this is inherently naive
\end{itemize}
- time complexity
\begin{itemize}
- training O(n*average\_doc\_length\_train+|c||dict|)
- testing O(|C| average\_doc\_length\_test)
\end{itemize}
- implementation
\begin{itemize}
- have symbol for unknown words
- underflow prevention - take logs of all probabilities so we don't get 0
- c = argmax log$P(c)$ + $\sum_i log P(X_i|c)$
\end{itemize}

## Instance-Based (ex. K nearest neighbors)
- also called lazy learners
- makes Voronoi diagrams
- can take majority vote of neighbors or weight them by distance
- distance can be Euclidean, cosine, or other
	- should scale attributes so large-valued features don't dominate
	- *Mahalanobois* distance metric takes into account covariance between neighbors
	- in higher dimensions, distances tend to be much farther, worse extrapolation
	- sometimes need to use *invariant metrics*
		- ex. rotate digits to find the most similar angle before computing pixel difference
			- could just augment data, but can be infeasible
		- computationally costly so we can approximate the curve these rotations make in pixel space with the *invariant tangent line*
			- stores this line for each point and then find distance as the distance between these lines
- finding NN with *k-d* (k-dimensional) tree
	- balanced binary tree over data with arbitrary dimensions
	- each level splits in one dimension
	- might have to search both branches of tree if close to split
- finding NN with *locality-sensitive hashing*
	- approximate
	- make multiple hash tables
		- each uses random subset of bit-string dimensions to project onto a line
		- union candidate points from all hash tables and actually check their distances
- comparisons
	- error rate of 1 NN is never more than twice that of Bayes error

# Feature Selection
## Filtering
- ranks features or feature subsets independently of the predictor
- univariate methods (consider one variable at a time)
\begin{itemize}
- ex. T-test of y for each variable
- ex. Pearson correlation coefficient - this can only capture linear dependencies
- mutual information - covers all dependencies
\end{itemize}
- multivariate methods
\begin{itemize}
- features subset selection
- need a scoring function
- need a strategy to search the space
-  sometimes used as preprocessing for other methods
\end{itemize}

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
\begin{itemize}
- symmetric D(A,B)=D(B,A)
- self-similarity D(A,A)=0
- positivity separation D(A,B)=0 iff A=B
- triangular inequality D(A,B) <= D(A,C)+D(B,C)
- ex. Minkowski Metrics $d(x,y)=\sqrt[r]{\sum |x_i-y_i|^r}$
\begin{itemize}
- r=1 Manhattan distance
- r=1 when y is binary -> Hamming distance
- r=2 Euclidean
- r=$\infty$ "sup" distance
\end{itemize}
- correlation coefficient - unit independent
- edit distance

## Hierarchical
- Two approaches:
    1. Bottom-up agglomerative clustering - starts with each object in separate cluster then joins
    2. Top-down divisive - starts with 1 cluster then separates
- ex. starting with each item in its own cluster, find best pair to merge into a new cluster
- repeatedly do this to make a tree (dendrogram)
- distances between clusters
\begin{itemize}
- single-link=nearest neighbor=their closest members (long, skinny clusters)
- complete-link=furthest neighbor=their furthest members (tight clusters)
- average=average of all cross-cluster pairs - most widely used
\end{itemize}
- Complexity: $O(n^2p)$ for first iteration and then can only get worse

## Partitional
- partition n objects into a set of K clusters (must be specified)
- globally optimal: exhaustively enumerate all partitions
- minimize sum of squared distances from cluster centroid
- Evaluation w/ labels - purity - ratio between dominant class in cluster and size of cluster

### Expectation Maximization (EM)
- general procedure that includes K-means
- E-step
\begin{itemize}
- calculate how strongly to which mode each data point “belongs” (maximize likelihood)
\end{itemize}
- M-step - calculate what each mode’s mean and covariance should be given the various responsibilities (maximization step)
- known to converge
- can be suboptimal
- monotonically decreases goodness measure
- can also partition around medoids
- mixture-based clustering \end{itemize}
\subsubsection{K-Means}
\begin{itemize}
- start with random centers
- assign everything to nearest center: O(|clusters|*np) 
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
- $L(\theta) = \sum_{i=1}^n (\hat{y}_i-y_i)^2+ \lambda ||\theta||_2^2$ 
- $L(\theta) = (X \theta - y)^T (X \theta -y)+ \lambda \theta^T \theta$
- $L(\theta) = \theta^T X^T X \theta - 2 \theta^T X^T y +y^T y +  \lambda \theta^T \theta$ 
- $0=\frac{\partial L}{\partial \theta} = 2X^TX\theta - 2X^T y+2\lambda \theta$
- $\theta = (X^TX+\lambda I)^{-1} X^T y$

## single Bernoulli
- L(p) = P(Train|Bernoulli(p)) = $P(X_1,...,X_n|p)=\prod_i P(X_i|p)=\prod_i p^{X_i} (1-p)^{1-X_i}$
- $=p^x (1-p)^{n-x}$ where x = $\sum x_i$
- $log(L(p)) = log(p^x (1-p)^{n-x}=x log(p) + (n-x) log(1-p)$
- $0=\frac{dL(p)}{dp} = \frac{x}{p} - \frac{n-x}{1-p} = \frac{x-xp - np+xp}{p(1-p)}=x-np$
- $\implies \hat{p} = \frac{x}{n}$

## multinomial
- $L(\theta)=P(Train|Multinomial(\theta))=P(d_1,...,d_n|\theta_1,...,\theta_p)$ where d is a document of counts x
- =$\prod_i^n P(d_i|\theta_1,...\theta_p)=\prod_i^n factorials \cdot \theta_1^{x_1},...,\theta_p^{x_p}$- ignore factorials because they are always same
\begin{itemize}
- require $\sum \theta_i = 1$
\end{itemize}
- $\implies \theta_i = \frac{\sum_{j=1}^n x_{ij}}{N}$ where N is total number of words in all docs