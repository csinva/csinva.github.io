---
layout: notes
title: Feature selection
category: ml
typora-copy-images-to: ./assets/ml
---

{:toc}

# filtering - select based on summary statistic

- ranks features or feature subsets independently of the predictor
- univariate methods (consider one variable at a time)
  - ex. variance threshold
  - ex. T-test of y for each variable
  - ex. correlation screening: pearson correlation coefficient - this can only capture linear dependencies
  - mutual information - covers all dependencies
  - ex. chi$^2$, f anova
- multivariate methods
  - features subset selection
  - need a scoring function
  - need a strategy to search the space
  - sometimes used as preprocessing for other methods

# wrapper - recursively eliminate features

- uses a predictor to assess features of feature subsets
- learner is considered a black-box - use train, validate, test set
- forward selection - start with nothing and keep adding
- backward elimination - start with all and keep removing
- others: Beam search - keep k best path at teach step, GSFS, PTA(l,r), floating search - SFS then SBS

# embedding - select from a model

- uses a predictor to build a model with a subset of features that are internally selected
- ex. lasso, ridge regression, random forest