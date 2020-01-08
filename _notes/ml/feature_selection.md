---
layout: notes
title: Feature selection
category: ml
typora-copy-images-to: ./assets/ml
---
* TOC
{:toc}


# filtering

- ranks features or feature subsets independently of the predictor
- univariate methods (consider one variable at a time)
  - ex. T-test of y for each variable
  - ex. pearson correlation coefficient - this can only capture linear dependencies
  - mutual information - covers all dependencies
- multivariate methods
  - features subset selection
  - need a scoring function
  - need a strategy to search the space
  - sometimes used as preprocessing for other methods

# wrapper

- uses a predictor to assess features of feature subsets
- learner is considered a black-box - use train, validate, test set
- forward selection - start with nothing and keep adding
- backward elimination - start with all and keep removing
- others: Beam search - keep k best path at teach step, GSFS, PTA(l,r), floating search - SFS then SBS

# embedding

- uses a predictor to build a model with a subset of features that are internally selected
- ex. lasso, ridge regression