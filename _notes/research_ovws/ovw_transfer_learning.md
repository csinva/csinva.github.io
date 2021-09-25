---
layout: notes
title: transfer learning
category: research
---


{:toc}


See also notes on causal inference for some close connections. 

# domain adaptation algorithms

*Domain test bed available [here](https://github.com/facebookresearch/DomainBed), for generalizating to new domains (i.e. performing well on domains that differ from previous seen data)*

- Empirical Risk Minimization (ERM, [Vapnik, 1998](https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034)) - standard training
- Invariant Risk Minimization (IRM, [Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893)) - learns a feature representation such that the optimal linear classifier on top of that representation matches across domains.
- Group Distributionally Robust Optimization (GroupDRO, [Sagawa et al., 2020](https://arxiv.org/abs/1911.08731)) - ERM + increase importance of domains with larger errors (see also papers from Sugiyama group e.g. [1](http://papers.neurips.cc/paper/3019-mixture-regression-for-covariate-shift.pdf), [2](https://arxiv.org/abs/1611.02041))
  - Variance Risk Extrapolation (VREx, [Krueger et al., 2020](https://arxiv.org/abs/2003.00688)) - encourages robustness over affine combinations of training risks, by encouraging strict equality between training risks
- Interdomain Mixup (Mixup, [Yan et al., 2020](https://arxiv.org/abs/2001.00677)) - ERM on linear interpolations of examples from random pairs of domains + their labels
- Marginal Transfer Learning (MTL, [Blanchard et al., 2011-2020](https://arxiv.org/abs/1711.07910)) - augment original feature space with feature vector marginal distributions and then treat as a supervised learning problem
- Meta Learning Domain Generalization (MLDG, [Li et al., 2017](https://arxiv.org/abs/1710.03463)) - use MAML to meta-learn how to generalize across domains
- learning more diverse predictors
  - Representation Self-Challenging (RSC, [Huang et al., 2020](https://arxiv.org/abs/2007.02454)) - adds dropout-like regularization to important features, forcing model to depend on many features
  - Spectral Decoupling (SD, [Pezeshki et al., 2020](https://arxiv.org/abs/2011.09468)) - regularization which forces model to learn more predictive features, even when only a few suffice
- embedding prior knowledge
  - Style Agnostic Networks (SagNet, [Nam et al., 2020](https://arxiv.org/abs/1910.11645)) - penalize style features (assumed to be spurious)
  - Penalizing explanations ([Rieger et al. 2020](https://arxiv.org/abs/1909.13584)) - penalize spurious features using prior knowledge
- Domain adaptation under structural causal models ([chen & buhlmann, 2020]((https://arxiv.org/abs/2010.15764)))
  - make clearer assumptions for domain adaptation to work
  - introduce CIRM, which works better when both covariates and labels are perturbed in target data
- kernel approach ([blanchard, lee & scott, 2011](https://papers.nips.cc/paper/2011/file/b571ecea16a9824023ee1af16897a582-Paper.pdf)) - find an appropriate RKHS and optimize a regularized empirical risk over the space



## domain invariance

*key idea: want repr. to be invariant to domain label*

- ![Screen Shot 2020-11-10 at 12.05.12 PM](../assets/domain_adv_training.png)
- same idea is used to [learn fair representations](https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf), but domain label is replaced with sensitive attribute
- Domain Adversarial Neural Network (DANN, [Ganin et al., 2015](https://arxiv.org/abs/1505.07818))
- Conditional Domain Adversarial Neural Network (CDANN, [Li et al., 2018](https://arxiv.org/abs/1807.08479)) - variant of DANN matching the conditional distributions  across domains, for all labels 
- Deep CORAL (CORAL, [Sun and Saenko, 2016](https://arxiv.org/abs/1607.01719)) - match mean / covariance of feature distrs
- Maximum Mean Discrepancy (MMD, [Li et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf))
- adversarial discriminative domain adaptation (ADDA [tzeng et al. 2017](https://arxiv.org/abs/1702.05464))
- balancing with [importance weighting](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.370.4921&rep=rep1&type=pdf)
- [Learning Robust Representations by Projecting Superficial Statistics Out](https://arxiv.org/abs/1903.06256) (wang et al. 2019)

## dynamic selection

*Dynamic Selection (DS) refers to techniques in which, for a new test point, pre-trained classifiers are selected/combined from a pool at test time  [review paper](https://www.etsmtl.ca/Unites-de-recherche/LIVIA/Recherche-et-innovation/Publications/Publications-2017/RCruz_InfoFusion.pdf) (cruz et al. 2018), [python package](https://github.com/scikit-learn-contrib/DESlib)*

1. define region of competence
   1. clustering
   2. kNN - more refined than clustering
   3. decision space - e.g. a model's classification boundary, internal splits in a model
   4. potential function - weight all the points (e.g. by their distance to the query point)
2. criteria for selection
   1. individual scores: acc, prob. behavior, rank, meta-learning, complexity
   2. group: data handling, ambiguity, diversity
3. combination
   1. non-trainable: mean, majority vote, product, median, etc.
   2. trainable: learn the combination of models
      1. related: in mixture of experts models + combination are trained jointly
   3. dynamic weighting: combine using local competence of base classifiers
   4. Oracle baseline - selects classifier predicts correct label, if such a classifier exists 

## test-time adaptation

- test-time adaptation
  - test-time augmentation
  - [batch normalization](https://arxiv.org/abs/1603.04779) (AdaBN)
  - [label shift estimation](https://arxiv.org/abs/1802.03916) (BBSE) - $p(y)$ shifts but $P(x|y)$ does not
  - [rotation prediction](https://arxiv.org/abs/1909.13231) (sun et al. 2020)
  - [entropy minimization](https://arxiv.org/abs/2006.10726) (test-time entropy minimization, TENT, wang et al. 2020) - optimize for model confidence (entropy of predictions), using only norm. statistics and channel-wise affine transformations

- combining train-time and test-time adaptation
  - Adaptive Risk Minimization (ARM, [Zhang et al., 2020](https://arxiv.org/abs/2007.02931)) - combines groups at training time + *batches at test-time*
    - *meta-train* the model using simulated distribution shifts, which is enabled by the training groups, such that it exhibits strong *post-adaptation* performance on each shift

# adv attacks

- [Adversarial Attacks and Defenses in Images, Graphs and Text: A Review](https://arxiv.org/abs/1909.08072) (xu et al. 2019) 
- attacks
  - [Barrage of Random Transforms for Adversarially Robust Defense](http://openaccess.thecvf.com/content_CVPR_2019/papers/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.pdf) (raff et al. 2019) 
  - [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/abs/1511.04599) (Moosavi-Dezfooli et. al 2016)
- defenses
  - a possible defense against adversarial attacks is to solve the anticausal classification problem by modeling the causal generative direction, a method which in vision is referred to as *analysis by synthesis* ([Schott et al., 2019](https://arxiv.org/abs/1805.09190))
- robustness vs accuracy
  - [robustness may be at odds with accuracy](https://openreview.net/pdf?id=SyxAb30cY7) (tsipiras...madry, 2019)
  - [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/abs/1901.08573) (Zhang, Jordan, et. al. 2019)
- adversarial examples
  - [Decision Boundary Analysis of Adversarial Examples ](https://pdfs.semanticscholar.org/08c5/88465b7d801ad912ef3e9107fa511ea0e403.pdf)(He, Li, & Song 2019)
  - [Natural Adversarial Examples](https://arxiv.org/abs/1907.07174) (Hendrycks, Zhao, Basart, Steinhardt, & Song 2020)
  - [Image-Net-Trained CNNs Are Biased Towards Texture](https://openreview.net/pdf?id=Bygh9j09KX) (Geirhos et al. 2019)
- transferability
  - [Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples](https://arxiv.org/abs/1605.07277) (papernot, mcdaniel, & goodfellow, 2016)
  - [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/pdf/1705.07204.pdf) (tramer et al. 2018)
  - [Improving Adversarial Robustness via Promoting Ensemble Diversity](https://arxiv.org/pdf/1901.08846.pdf) (pang et al. 2019)
    - encourage diversity in non-maximal predictions
- ranking
  - [Automatically Discovering and Learning New Visual Categories with Ranking Statistics](https://arxiv.org/pdf/2002.05714.pdf)
- adversarial training: $$\min _{\boldsymbol{\theta}} \frac{1}{N} \sum_{n=1}^{N} \operatorname{Loss}\left(f_{\theta}\left(x_{n}\right), y_{n}\right)+\lambda\left[\max _{\|\delta\|_{\infty} \leq \epsilon} \operatorname{Loss}\left(f_{\theta}\left(x_{n}+\delta\right), y_{n}\right)\right]$$
- robustness as a constraint not a loss ([Constrained Learning with Non-Convex Losses](https://arxiv.org/abs/2103.05134) (chamon et al. 2021))
  - $$\begin{aligned}
    \min _{\boldsymbol{\theta}} & \frac{1}{N} \sum_{n=1}^{N} \operatorname{Loss}\left(f_{\theta}\left(x_{n}\right), y_{n}\right) \\
    \text { subject to } & \frac{1}{N} \sum_{n=1}^{N}\left[\max _{\|\delta\|_{\infty} \leq \epsilon} \operatorname{Loss}\left(f_{\theta}\left(\boldsymbol{x}_{n}+\delta\right), y_{n}\right)\right] \leq c
    \end{aligned}$$
  - when penalty is convex, these 2 problems are the same