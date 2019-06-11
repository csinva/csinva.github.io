---
layout: notes
section-type: notes
title: causal inference
category: stat
---

* TOC
{:toc}
---

# causality (freedman ch 1)

- When using observational (non-experimental) data to make causal inferences, the key problem is *confounding*
  - *stratification* = *cross-tabulation* - only look at when confounding variables have same value
- association is circumstantial evidence for causation
- examples
  - HIP trial of mammography - want to do whole treatment group v. whole control group
  - Snow on cholera - water
  - causes of poverty - Yul's model, changes with lots of things
- problem: never get to see gt

# basic causal inference

- *confounding* - difference between groups other than the treatment which affects the response
- [bradford hill criteria](https://en.wikipedia.org/wiki/Bradford_Hill_criteria) - some simple criteria for establishing causality (e.g. strength, consistency, specificity)
- 3 frameworks
  1. neyman-rubin model: $Y_i = T_i a_i + (1-T_i) b_i$
    - $\hat{ate} = \hat{a}_A - \hat{b}_B$
    - $\hat{ate}_{adj} = [\bar{a}_A - (\bar{x}_A - \bar{x})^T \hat{\theta}_A] - [\bar{b}_B - (\bar{x}_B - \bar{x})^T \hat{\theta}_B]$
      - $\hat{\theta}_A = argmin \sum_{i \in A} (a_i - \bar{a}_A - (x_i - \bar{x}_A)^T \theta)^2$

  2. neyman-pearson
    - null + alternative hypothesis
      - null is favored unless there is strong evidence to refute it
  3. fisherian testing framework
    - small p-values evidence against null hypothesis
    - null hypothesis
- errors
  - *type I err*: FP - reject when false
  - *type II err*: FN
  - *power*: TP = sensitivity
  - TN
  - newer
    - sensitivity = power
    - recall = sensitivity - true positive rate = TP / P
    - precision = TP / (TP + FP)
    - specificity = true neg rate = TN / N
- natural experiments
  - ex. john snow
- *propensity score* - probability that a subject recieving a treatment is valid after conditioning on appropriate covariates
- 3 principles of experimental design
  1. replication
  2. randomization
  3. conditioning

# 2 general approaches

1. matching - find patients that are similar and differ only in the treatment
   1. only variables you don't match on could be considered causal
2. regression
   - requires *unconfoundedness* = *omitted variable bias*
   - if there are no confounders, correlation is causation

# causal inference papers

- Hainmueller & Hangartner (2013) - Swiss passport
  - naturalization decisions vary with immigrants' attributes
  - is there immigration against immigrants based on country of origin?
  - citizenship requires voting by municipality
- Sekhon et al. - when natural experiments are neither natural nor experiments
  - even when natural interventions are randomly as- signed, some of the treatment–control comparisons made available by natural experiments may not be valid
- Grossman et al. - "Descriptive Representation and Judicial Outcomes in Multiethnic Societies"
  - judicial outcomes of arabs depended on whether there was an Arab judge on the panel
- liver transplant
  - maximize benefit (life with - life without)
  - currently just goes to person who would die quickest without
  - Y = T Y(1) + (1-T) Y(0)
    - Y(1) = survival with transplant
    - Y(0) = survival w/out transplant
      - fundamental problem of causal inference - can 't observe Y(1) and Y(0)
    - T = 1 if receive transplant else 0
  - goal: estimate $\tau = Y(1) - Y(0)$ for each person

# causality ovw

![Screen Shot 2019-04-07 at 7.01.55 PM](assets/Screen Shot 2019-04-07 at 7.01.55 PM.png)



# pearl's causal ladder

- [blog post](http://smithamilli.com/blog/causal-ladder/)
  - prediction - just need to have the joint distr. of all the variables
  - intervention - we can change things and get conditionals based on evidence **after intervention**
  - counterfactuals - we can change things and get conditionals based on evidence **before intervention**

# bottou causality

- ex. [lopez-paz_17](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lopez-Paz_Discovering_Causal_Signals_CVPR_2017_paper.pdf)
  - C(A, B) - count number of images in which B would disappear if A was removed
  - we say A *causes* B when C(A, B) is (sufficiently) greater than the converse C(B, A)
  - basics
    - given joint distr. of (A, B), we want to know if A -> B, B-> A
      - with no assumptions, this is nonidentifiable
    - requires 2 assumptions
      - ICM: independence between cause and mechanism (i.e. the function doesn't change based on distr. of X) - this usually gets violated in anticausal direction
      - causal sufficiency - we aren't missing any vars
    - ex. ![Screen Shot 2019-05-20 at 10.04.03 PM](assets/Screen Shot 2019-05-20 at 10.04.03 PM.png)
      - here noise is indep. from x (causal direction), but can't be independent from y (non-causal direction)
      - in (c), function changes based on input
    - can turn this into binary classification and learn w/ network: given X, Y, does X->Y or Y-X?
  - on images, they get scores for different objects (w/ bounding boxes)
    - eval - when one thing is erased, does the other also get erased?
- [link to iclr talk](https://www.technologyreview.com/s/613502/deep-learning-could-reveal-why-the-world-works-the-way-it-does/?fbclid=IwAR3LF2dc_3EvWXzEHhtrsqtH9Vs-4pjPALfuqKCOma9_gqLXMKDeCWrcdrQ)
- [visual causal feature learning](https://arxiv.org/abs/1412.2309)
- [The Hierarchy of Stable Distributions and Operators to Trade Off Stability and Performance](https://arxiv.org/abs/1905.11374)
  - different predictors learn different things
  - only pick the stable parts of what they learn (in a graph representation)