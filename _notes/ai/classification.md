---
layout: notes
section-type: notes
title: Classification
category: ai
---
* TOC
{:toc}


[TOC]


# Classification
- asymptotic classifier - assume you get infinite training / testing points
- *discriminative* - model $P(C\vert X)$ directly
  - smaller asymptotic error
  - slow convergence ~ O(p)
- *generative* - model $P(X\vert C)$  directly
  - generally has higher bias -> can handle missing data
  - fast convergence ~ O(log(p))


## binary classification
- $\hat{y}=sign(w^T x)$
- usually don't minimize 0-1 loss (combinatorial)
- usually $w^Tx$ includes b term, but generally we don't want to regularize b

| Model               | $\hat{w}$ objective                      |
| ------------------- | ---------------------------------------- |
| Perceptron          | $\sum_i max(0,  -y_i \cdot w^T x_i)$     |
| Linear SVM          | $w^Tw + C \sum_i max(0,1-y_i \cdot w^T x_i)$ |
| Logistic regression | $w^Tw + C \sum_i log(1+exp(-y_i \cdot w^T x_i))$ |
- *perceptron* - tries to find separating hyperplane
  - whenever misclassified, update w
  - can add in delta term to maximize margin

## multiclass classification

- reducing multiclass (K categories) to binary
  - *one-against-all*
    - train K binary classifiers
    - class i = positive otherwise negative
    - take max of predictions
  - *one-vs-one* = *all-vs-all*
    - train C(K,2) binary classifiers
    - labels are class i and class j
    - inference - any class can get up to k-1 votes, must decide how to break ties
  - flaws - learning only optimizes *local correctness*
- single classifier
  - *multiclass perceptron* (Kesler)
    - if label=i, want $w_i ^Tx > w_j^T x \quad \forall j$
    - if not, update $w_i$ and $w_j$* accordingly
    - *kessler construction*
      - $w =  [w_1 ... w_k] $
      - want $w_i^T x > w_j^T x \quad \forall j$
      - rewrite $w^T \phi (x,i) > w^T \phi (x,j) \quad \forall j$
        - here $\phi (x,i)$ puts x in the ith spot and zeros elsewhere
        - $\phi$ is often used for feature representation
      - define margin: 
        $\Delta (y,y') = \begin{cases} 
        \delta& if y \neq y' \\\
        0& if y=y'
        \end{cases}$
      - check if $y=argmax_{y'}w^T \phi(x,y') + \delta (y,y')$
  - multiclass SVMs (Crammer&Singer)
    - minimize total norm of weights s.t. true label is score at least 1 more than second best label
  - multinomial logistic regression = multi-class *log-linear* model
    - $P(y\vert x,w)=\frac{exp(w^T_yx)}{\sum_{y' \in \{ 1,...,K\}} exp(w_{y'}^T,x)}$
      - we control the peakedness of this by dividing by stddev
    - *soft-max*: sometimes substitue this for $w^T_y x$ 

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
- *geometric margin* = functional margin / $\vert \vert w\vert \vert $
  - if $\vert \vert w\vert \vert =1$, then same as functional margin
  - invariant to scaling of w
- optimal margin classifier
  - want $$max \: \gamma \: s.t. \: y^{(i)} (w^T x^{(i)} + b) \geq \gamma, i=1,..,m; \vert \vert w\vert \vert =1$$
    - difficult to solve, especially because of $\vert \vert w\vert \vert =1$ constraint
    - assume $\hat{\gamma}=1$ - just a scaling factor
    - now we are maximizing $1/\vert \vert w\vert \vert $
  - equivalent to this formulation: $$min \: 1/2 \vert \vert w\vert \vert ^2 \: s.t. \: y^{(i)}(w^Tx^{(i)}+b)\geq1, i = 1,...,m$$
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
  - we get this from $M=\vert x^+ - x^-\vert  = \vert \lambda w\vert  = \lambda \sqrt{w^Tw} $
  - separable case: argmin($w^Tw$) subject to 
  - $w^Tx+b\geq 1$ for all x in +1 class
  - $w^Tx+b\leq 1$ for all x in -1 class
- solve with quadratic programming
- non-separable case: argmin($w^T w/2 + C \sum_i^n \epsilon_i$) subject to
  - $w^Tx_i +b \geq 1-\epsilon_i $ for all x in +1 class
  - $w^Tx_i +b \leq -1+\epsilon_i $ for all x in -1 class
  - $\forall i, \epsilon_i \geq 0$
  - large C can lead to overfitting
- benefits
  - number of parameters remains the same (and most are set to 0)
  - we only care about support vectors
  - maximizing margin is like regularization: reduces overfitting
- these can be solved with quadratic programming QP
- solve a dual formulation (Lagrangian) instead of QPs directly so we can use kernel trick
  - primal: $min_w max_\alpha L(w,\alpha)$
  - dual: $max_\alpha min_w L(w,\alpha)$
- KKT condition for strong duality
  - complementary slackness: $\lambda_i f_i(x) = 0, i=1,...,m$
- VC (Vapnic-Chervonenkis) dimension - if data is mapped into sufficiently high dimension, then samples will be linearly separable (N points, N-1 dims)
- kernel functions - new ways to compute dot product (similarity function)
  - original testing function: $\hat{y}=sign(\Sigma_{i\in train} \alpha_i y_i x_i^Tx_{test}+b)$
  - with kernel function: $\hat{y}=sign(\Sigma_{i\in train} \alpha_i y_i K(x_i,x_{test})+b)$
  - linear $K(x,z) = x^Tz$
  - polynomial $K (x, z) = (1+x^Tz)^d$
  - radial basis kernel $K (x, z) = exp(-r\vert \vert x-z\vert \vert ^2)$
  - computing these is O($m^2$), but dot-product is just O(m)
  - function that corresponds to an inner product in some expanded feature space
- practical guide
  - use m numbers to represent categorical features
  - scale before applying
  - fill in missing values
  - start with RBF

### Logistic Regression
- $p_i = P(Y_i=1\|x_i) = exp(x_i^T \theta) / (1+exp(x_i^T \theta))$
- $Logit(p_i) = log(p_i / (1-p_i)) = x_i^T \theta$
- predict using Bernoulli distribution with this parameter p
- can be extended to multiple classes - multinomial distribution

## Generative

### Naive Bayes Classifier
- let $C_1,...,C_L$ be the classes of Y
- want Posterior $P(C\vert X) = \frac{P(X\vert C)(P(C)}{P(X)}$ 
- MAP rule - maximum A Posterior rule
  \begin{itemize}
- use Prior P(C)
- using x, predict $C^*=\text{argmax}_C P(C\vert X_1,...,X_p)=\text{argmax}_C P(X_1,...,X_p\vert C) P(C)$ - generally ignore denominator
  \end{itemize}
- naive assumption - assume that all input attributes are conditionally independent given C
  \begin{itemize}
- $P(X_1,...,X_p\vert C) = P(X_1\vert C)\cdot...\cdot P(X_p\vert C) = \prod_i P(X_i\vert C)$ 
  \end{itemize}
- learning
  \begin{enumerate}
- learn L distributions $P(C_1),P(C_2),...,P(C_L)$
- learn $P(X_j=x_{jk}\vert C_i)$ 
  \begin{itemize}
- for j in 1:p
- i in 1:$\vert C\vert $
- k in 1:$\vert X_j\vert $
- for discrete case we store $P(X_j\vert c_i)$, otherwise we assume a prob. distr. form
  \end{itemize}
  \begin{itemize}
- naive: $\vert C\vert  \cdot (\vert X_1\vert  + \vert X_2\vert  + ... + \vert X_p\vert )$ distributions
- otherwise: $\vert C\vert \cdot (\vert X_1\vert  \cdot \vert X_2\vert  \cdot ... \cdot \vert X_p\vert )$
  \end{itemize}
  \end{enumerate}
- testing
  \begin{itemize}
- $P(X\vert c)$ - look up for each feature $X_i\vert C$ and try to maximize
  \end{itemize}
- smoothing - used to fill in 0s
  \begin{itemize}
- $P(x_i\vert c_j) = \frac{N(x_i, c_j) +1}{N(c_j)+\vert X_i\vert }$ 
- then, $\sum_i P(x_i\vert c_j) = 1$
  \end{itemize}

### Gaussian classifiers
- distributions
  \begin{itemize}
- Normal $P(X_j\vert C_i) = \frac{1}{\sigma_{ij} \sqrt{2\pi}} exp\left( -\frac{(X_j-\mu_{ij})^2}{2\sigma_{ij}^2}\right)$- requires storing $\vert C\vert \cdot p$ distributions
- Multivariate Normal $\frac{1}{(2\pi)^{D/2}} \frac{1}{\vert \Sigma\vert ^{1/2}} exp\left(-\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu)\right)$where $\Sigma$ is covariance matrix
  \end{itemize}
  \end{itemize}
  \begin{itemize}
- decision boundary are points satisfying $P(C_i\vert X) = P(C_j\vert X)$
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
- multivariate Bernoulli: $P(X\vert C)=P(w_1=true,w_2=false,...\vert C)$
- multivariate Binomial: $P(X\vert C)=P(w_1=n_1,w_2=n_2,...\vert C)$
  \begin{itemize}
- this is inherently naive
  \end{itemize}
- time complexity
  \begin{itemize}
- training O(n*average\_doc\_length\_train+$\vert c\vert \vert dict\vert $)
- testing O($\vert C\vert $ average\_doc\_length\_test)
  \end{itemize}
- implementation
  \begin{itemize}
- have symbol for unknown words
- underflow prevention - take logs of all probabilities so we don't get 0
- c = argmax log$P(c)$ + $\sum_i log P(X_i\vert c)$
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