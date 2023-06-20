---
layout: notes
title: uncertainty
category: research
---

**some notes on uncertainty in machine learning, particularly deep learning**

{:toc}

# basics

- **calibration** - predicted probabilities should match real probabilities
  - platt scaling - given trained classifier and new calibration dataset, basically just fit a logistic regression from the classifier predictions -> labels
  - isotonic regression - nonparametric, requires more data than platt scaling
    - piecewise-constant non-decreasing function instead of logistic regression
- **confidence** - predicted probability = confidence, max margin, entropy of predicted probabilities across classes
- **ensemble uncertainty** - ensemble predictions yield uncertainty (e.g. variance within ensemble)
- **quantile regression** - use quantile loss to penalize models differently + get confidence intervals
  - [can easily do this with sklearn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)
  - quantile loss = $\begin{cases} \alpha \cdot \Delta & \text{if} \quad \Delta > 0\\\\(\alpha - 1) \cdot \Delta & \text{if} \quad \Delta < 0\end{cases}$
    - $\Delta =$ actual - predicted
    - ![Screen Shot 2019-06-26 at 10.06.11 AM](../assets/quantile_losses.png)
  - [Single-Model Uncertainties for Deep Learning](https://arxiv.org/abs/1811.00908) (tagovska & lopez-paz 2019) - use simultaneous quantile regression



# outlier-detection

*Note: outlier detection uses information only about X to find points "far away" from the main distribution*

- overview from [sklearn](https://scikit-learn.org/stable/modules/outlier_detection.html)
  - **elliptic envelope** - assume data is Gaussian and fit elliptic envelop (maybe robustly) to tell when data is an outlier
  - **local outlier factor** (breunig et al. 2000) - score based on nearest neighbor density
  - idea: gradients should be larger if you are on the image manifold
  - [isolation forest](https://ieeexplore.ieee.org/abstract/document/4781136) (liu et al. 2008) - lower average number of random splits required to isolate a sample means more outlier
  - [one-class svm](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM) - estimates the support of a high-dimensional distribution using a kernel (2 approaches:)
    - separate the data from the origin (with max margin between origin and points) (scholkopf et al. 2000)
    - find a sphere boundary around a dataset with the volume of the sphere minimized ([tax & duin 2004](https://link.springer.com/article/10.1023/B:MACH.0000008084.60811.49))

- [detachment index](https://escholarship.org/uc/item/9d34m0wz) (kuenzel 2019) - based on random forest
  - for covariate $j$, detachment index $d^j(x) = \sum_i^n w (x, X_i) \vert X_i^j - x^j \vert$
    - $w(x, X_i) = \underbrace{1 / T\sum_{t=1}^{T}}_{\text{average over T trees}} \frac{\overbrace{1\{ X_i \in L_t(x) \}}^{\text{is }   X_i \text{ in the same leaf?}}}{\underbrace{\vert L_t(x) \vert}_{\text{num points in leaf}}}$ is $X_i$ relevant to the point $x$?

# uncertainty detection

*Note: uncertainty detection uses information about X / $\phi(X)$ and Y,  to find points for which a particular prediction may be uncertain. This is similar to the predicted probability output by many popular classifiers, such as logistic regression.*

- **rejection learning** - allow models to *reject* (not make a prediction) when they are not confidently accurate ([chow 1957](https://ieeexplore.ieee.org/abstract/document/5222035/?casa_token=UiIdn8AjFjYAAAAA:XvnZPA7rJlvwxD-bIh2dNG4SPfnHtDYWcBUmAFYRxD6Xk8QE5osnKLs8tAlib_doL8OxqYjMLDE), [cortes et al. 2016](https://link.springer.com/chapter/10.1007/978-3-319-46379-7_5))
- [To Trust Or Not To Trust A Classifier](http://papers.nips.cc/paper/7798-to-trust-or-not-to-trust-a-classifier.pdf) (jiang, kim et al 2018) - find a trusted region of points based on nearest neighbor density (in some embedding space)
  - trust score uses density over some set of nearest neighbors
  - do clustering for each class - trust score = distance to once class's cluster vs the other classes

## bayesian approaches

- **epistemic uncertainty** - uncertainty in the DNN model parameters
  - without good estimates of this, often get aleatoric uncertainty wrong (since $p(y\vert x) = \int p(y \vert x, \theta) p(\theta \vert data) d\theta$
- **aleatoric uncertainty** -  inherent and irreducible data noise (e.g. features contradict each other)
  - this can usually be gotten by predicting a distr. $p(y \vert x)$ instead of a point estimate
  - ex. logistic reg. already does this
  - ex. regression - just predict mean and variance of Gaussian
- [gaussian processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)



# neural networks

## ensembles

- [DNN ensemble uncertainty works](http://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles) - predict mean and variance w/ each network then ensemble (don't need to do bagging, random init is enough)
- [Deep Ensembles: A Loss Landscape Perspective](https://arxiv.org/abs/1912.02757v1) (fort, hu, & lakshminarayanan, 2020)
  - different random initializations provide most diversity
  - samples along one path have varying weights but similar predictions
  - ![deep_ensembles](../assets/deep_ensembles.png)
- [Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning](https://github.com/bayesgroup/pytorch-ensembles) - many complex ensemble approaches are similar to just an ensemble of a few randomly initialized DNNs

## directly predict uncertainty

- [Inhibited Softmax for Uncertainty Estimation in Neural Networks](https://arxiv.org/abs/1810.01861) (mozejko et al. 2019) - directly predict uncertainty by adding an extra output during training
- [Learning Confidence for Out-of-Distribution Detection in Neural Networks](https://arxiv.org/pdf/1802.04865.pdf) (devries et al. 2018) - predict both prediction *p* and confidence *c*
  - during training, learn using $p' = c \cdot p + (1 - c) \cdot y$
- [Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers](https://arxiv.org/abs/1805.08206) (geifmen et al. 2019)
    - just predicting uncertainty is biased
    - estimate uncertainty of highly confident points using earlier snapshots of the trained model
- [Contextual Outlier Interpretation](https://arxiv.org/abs/1711.10589) (liu et al. 2018) - describe outliers with 3 things: outlierness score, attributes that contribute to the abnormality, and contextual description of its neighborhoods
    - [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759) (liu et al. 2021)
- [Getting a CLUE: A Method for Explaining Uncertainty Estimates](https://arxiv.org/abs/2006.06848) 
- [The Right Tool for the Job: Matching Model and Instance Complexities - ACL Anthology](https://aclanthology.org/2020.acl-main.593/) - at each layer, model outputs a prediction - if it's confident enough it returns, otherwise it continues on to the next layer

## nearest-neighbor methods

- [Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning](https://arxiv.org/pdf/1803.04765.pdf) (papernot & mcdaniel, 2018)
- [distance-based confidence scores](https://arxiv.org/pdf/1709.09844.pdf) (mandelbaum et al. 2017) - use either distance in embedding space or adversarial training to get uncertainties for DNNs
- [deep kernel knn](https://arxiv.org/pdf/1811.02579.pdf) (card et al. 2019) - predict labels based on weighted sum of training instances, where weights are given by distance in embedding space
    - add an uncertainty based on conformal methods

## bayesian neural networks

- [blog posts on basics](https://medium.com/neuralspace/probabilistic-deep-learning-bayes-by-backprop-c4a3de0d9743)
  - want $p(\theta|x) = \frac {p(x|\theta) p(\theta)}{p(x)}$
    - $p(x)$ is hard to compute
- [slides on basics](https://wjmaddox.github.io/assets/BNN_tutorial_CILVR.pdf)
- [Bayes by backprop (blundell et al. 2015)](https://arxiv.org/abs/1505.05424) - efficient way to train BNNs using backprop
  - Instead of training a single network, trains an ensemble of networks, where each network has its weights drawn from a shared, learned probability distribution. Unlike other ensemble methods, the method typically only doubles the number of parameters yet trains an infinite ensemble using unbiased Monte Carlo estimates of the gradients.
- [Evaluating Scalable Bayesian Deep Learning Methods for Robust Computer Vision](https://arxiv.org/pdf/1906.01620.pdf)
- [icu bayesian dnns](https://aiforsocialgood.github.io/icml2019/accepted/track1/pdfs/38_aisg_icml2019.pdf)
  - focuses on epistemic uncertainty
  - could use one model to get uncertainty and other model to predict
- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](http://proceedings.mlr.press/v48/gal16.pdf)  
  - dropout at test time gives you uncertainty
- [SWAG](https://papers.nips.cc/paper/9472-a-simple-baseline-for-bayesian-uncertainty-in-deep-learning.pdf) (maddox et al. 2019) - start with pre-trained net then get Gaussian distr. over weights by training with large constant setp-size
- [Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors](https://arxiv.org/abs/2005.07186) (dusenberry, jerfel et al. 2020) - BNNs scale to SGD-level with better calibration



# conformal inference

- conformal inference constructs valid (wrt coverage error) prediction bands for individual forecasts
  - relies on few parametric assumptions
  - holds in finite samples for any distribution of (X, Y) and any algorithm $\hat f$
  - starts with vovk et al. '90
- simple example: construct a 95% interval for a new sample (not mean) by just looking at percentiles of the empirical data
  - empirical data tends to undercover (since empirical residuals tend to underestimate variance) - conformal inference aims to rectify this
- [Uncertainty Sets for Image Classifiers using Conformal Prediction](https://arxiv.org/abs/2009.14193) (angelopoulos, bates, malik, jordan, 2021)
  - Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging ([Angelopoulos, ...jordan, malik, upadhyayula, roman, '22](https://arxiv.org/pdf/2202.05265.pdf))
    - pixel-level uncertainties



# large language models (llms)

- Teaching Models to Express Their Uncertainty in Words ([Lin et al., 2022](https://arxiv.org/abs/2205.14334)) - GPT3 can  generate both an answer and a level of confidence (e.g. "90% confidence")
