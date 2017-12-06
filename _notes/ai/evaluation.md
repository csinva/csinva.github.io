---
layout: notes
section-type: notes
title: Evaluation
category: ai
---
* TOC
{:toc}

[toc]


# metrics
- Define a loss function $\mathcal{L}$
	- 0-1 loss: $\vert C-f(X)\vert$ 
	- $L_2$ loss: $(C-f(X))^2$
- Expected Prediction Error EPE(f) = $E_{X,C} [\mathcal{L}(C,f(X))]$
	- =$E_{X}\left[ \sum_i \mathcal{L}(C_i,f(X)) Pr(C_i\vert X) \right]$
- Minimize EPE
	- Bayes Classifier minimizes 0-1 loss: $\hat{f}(X)=C_i$ if $P(C_i\vert X)=max_f P(f\vert X)$
	- KNN minimizes L2 loss: $\hat{f}(X)=E(Y\vert X)$ 
- EPE(f(X)) = $noise^2+bias^2+variance$
	- noise - unavoidable
	- bias=$E[(\bar{\theta}-\theta_{true})^2]$ - error due to incorrect assumptions
		- simple linear regression has 0 bias
	- variance=$E[(\bar{\theta}-\theta_{estimated})^2]$ - error due to variance of training samples
- more complex models (more nonzero parameters) have lower bias, higher variance
	- if high bias, train and test error will be very close (model isn't complex enough)
- accuracy = number of correct classifications / total number of test cases
- balanced accuracy = 1/2 (TP/P + TN/N)
- recall - TP/(TP+FN)
- precision - TP/(TP+FP)
- you train by lowering SSE or MSE on training data
	- report MSE for test samples
	
# cv
- *cross validation* - don't have enough data for a test set
	- properties
		- not good when n < complexity of predictor
		- because summands are correlated
		- assume data units are exchangeable
		- can sometimes use this to pick k for k-means
		- data is reused
	- types
		1. k-fold - split data into N pieces
			- N-1 pieces for fit model, 1 for test
			- cycle through all N cases
			- average the values we get for testing
		2. leave one out (LOOCV)
			- train on all the data and only test on one
			- then cycle through everything
		3. random split - shuffle and repeat
		4. *one-way CV* = *prequential analysis* - keep testing on next data point, updating model
		5. ESCV - penalize variance between folds
- *regularization path* of a regression - plot each coeff v. $\lambda$
	- tells you which features get pushed to 0 and when
	
# stability
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
		3. *jackknife resampling*
			- subsample containing all but one of the points
