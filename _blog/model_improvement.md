---
layout: notes_without_title
section-type: notes
title: tips to improve a model
category: blog
---



# Tips to improve a model

**chandan singh** 

---

## data splitting

- is there any dependence within data splits (e.g. temporal correlations) that would artificially effect your accuracy estimates?

## visualizing data

- look at histograms of outcomes / key features
- see if features can be easily reduced to lower dimensions (e.g. PCA, LDA)

## preprocessing

- normalizing features and output
- balance the data (random sampling, random sampling + ensembling, [smote](https://jair.org/index.php/jair/article/view/10302), etc.)
- do [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) with simple screening (e.g. variance, correlation, etc)
- do feature selection using a model (e.g. tree, lasso)

## debugging

- can the model achieve zero training error on a single example?
- how do simple baselines (e.g. linear models, decision trees) perform?

## feature engineering

- visualize the outputs of sparse coding on your features

## modeling

- try ensembling

## feature importances

- do feature importances match what you would expect?

## analyzing erros

- plot predictions vs groundtruth
- try to [detect outliers](https://scikit-learn.org/stable/modules/outlier_detection.html)
- visualize the examples with the largest prediction error
