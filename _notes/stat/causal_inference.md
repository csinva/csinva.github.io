---
layout: notes
section-type: notes
title: causal inference
category: stat
---

* TOC
{:toc}
---

*Some notes on causal inference both from introductory courses following the neyman-rubin school of thought and based on Judea Pearl's ladder of causality*

# basics

- [good overview](https://arxiv.org/abs/1907.07271)
- when using observational (non-experimental) data to make causal inferences, the key problem is **confounding** - difference between groups other than the treatment which affects the response
  - *stratification* = *cross-tabulation* - only look at when confounding variables have same value
- [bradford hill criteria](https://en.wikipedia.org/wiki/Bradford_Hill_criteria) - some simple criteria for establishing causality (e.g. strength, consistency, specificity)
- association is circumstantial evidence for causation
- problem: never get to see gt
- groundtruth: randomized control trial (RCT) - controls for any possible confounders


## 2 general approaches

1. matching - find patients that are similar and differ only in the treatment
   1. only variables you don't match on could be considered causal
2. regression adjustments
   - requires *unconfoundedness* = *omitted variable bias*
   - if there are no confounders, correlation is causation

## common examples

- HIP trial of mammography - want to do whole treatment group v. whole control group
- Snow on cholera - water
- causes of poverty - Yul's model, changes with lots of things
- liver transplant
  - maximize benefit (life with - life without)
  - currently just goes to person who would die quickest without
  - Y = T Y(1) + (1-T) Y(0)
    - Y(1) = survival with transplant
    - Y(0) = survival w/out transplant
      - fundamental problem of causal inference - can 't observe Y(1) and Y(0)
    - T = 1 if receive transplant else 0
  - goal: estimate $\tau = Y(1) - Y(0)$ for each person

# potential outcome framework (neyman-rubin)

- advantages over DAGs
  - easy to express some common assumptions, such as monotonicity / convexity
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
- natural experiments
  - ex. john snow
- *propensity score* - probability that a subject recieving a treatment is valid after conditioning on appropriate covariates
- 3 principles of experimental design
  1. replication
  2. randomization
  3. conditioning

# causality DAGs (Pearl et al.)

![Screen Shot 2019-04-07 at 7.01.55 PM](assets/Screen Shot 2019-04-07 at 7.01.55 PM.png)



- more from the book of why
- advantages over potential outcomes
  - easy to express assumptions on what is independent, particularly when there are many variables
  - do-calculus allows for answering some specific questions easily
- [blog post on causal ladder](http://smithamilli.com/blog/causal-ladder/)
- [intro to do-calculus post](https://www.inference.vc/untitled/) and subsequent posts

## 1- **prediction/association** - just need to have the joint distr. of all the variables

- basically just $p(y|x)$

## 2 - **intervention** - we can change things and get conditionals based on evidence **after intervention**

- $p(y|do(x))$ - which represents the conditional distr. we would get if we were to manipulate $x$ in a randomized trial
  - to get this, we assume the causal structure (can still kind of test it based on conditional distrs., can sometimes use causal discovery techniques to try to identify the causal diagram under just some smoothness / independence assumptions)
  - having assumed the structure, we delete all edges going into a do operator and set the value of $x$
  - then, do-calculus yields a formula to estimate $p(y|do(x))$ assuming this causal structure
    - 3 rules which go from do-calculus to probability expressiom (remove do operator from statement and allow us to calculate it)
  - see introductory paper [here](https://arxiv.org/pdf/1305.5506.pdf), more detailed paper [here](https://ftp.cs.ucla.edu/pub/stat_ser/r416-reprint.pdf) (pearl 2013)
- by assuming structure, we learn how large impacts are

## 3 - **counterfactuals** - we can change things and get conditionals based on evidence **before intervention**
- probablilistic answer to a "what would have happened if" question
- very similar to neyman's potential outcome framework
- simple matching is often not sufficient (need a very good model for how to match, hopefully a causal one)
- this is for a specific data point, not a randomly sampled data point like an intervention would be
  - instead of intervention $p(y|do(x))$ we get $p(y^*|x^*, z=z)$ where z represents fixing all the other variables and $y^*$ and $x^*$ are not observed
  - averaging over all data points, we'd expect to get something similar to the intervention $p(y|do(x))$
- requires SEM - structured equation model not just causal graph
  - this is set of equations which tell how to compute value of each node given parents (and maybe some noise $\epsilon$ for each node)
  - again, fix value of $x$ (and values of $\epsilon$ seend in the data) and use SEM to set all downstream variables
- this allows for our intervention to contradict something we condition on 
	- 	e.g. "Given that Hillary lost and didn't visit Michigan, would she win if she had visited Michigan?"
	- 	e.g. “What fraction of patients who are treated and died would have survived if they were not treated?”
- 	exogenous nodes - node in the network that represents all the data note collected

## technical notes

- **case-control study** - retrospective - compares "cases" (people with a disease) to controls
- **sensitivity analysis** - instead of drawing conclusions by assuming the absence of certain causal relationships, challenge such assumptions and evaluate how strong altervnative relationships must be in order to explain the observed data
- **regression-based adjustment** - if we know the confounders, can just regress on the confounders and the treatment and the coefficient for the treatment (the partial regression coefficient) will give us the average causal effect)
  - works only for linear models
- **back-door criterion** - want to deconfound 2 variables X and Y: http://bayes.cs.ucla.edu/BOOK-2K/ch3-3.pdf
  - ensure that there is no path which points to X which allows dependence between X and Y ( paths which point to X are non-causal, representing confounders )
  - remember, in DAG junctions conditioning makes things independent unless its at a V junction
- **front-door criterion** - want to deconfound treatment from outcome, even without info on the confounder
  - only really need to know about treatment, M, and outcome

```mermaid
graph LR
C(Confounder) -->Y(Outcome)
C --> X(Treatment)
X --> M
M --> Y
```

- **instrumental variables** - variable which can be used to effectively due a RCT because it was made random by some external factor
  - ex. army draft, john snow's cholera study

## historical notes

- key problem: no language to write causal relationships 
- counter factual: set value but erase all arrows going into the variable which we set (everything else the same)
- do causal diagrams exist in brain?
- regression to the mean - galton and pearson originally discover correlation instead of causation
- sewall wright studying guinea pigs uses causation to predict correlations (path analysis)
- path analysis became structural equation modeling

## paradox examples

- monty hall problem: why you should switch
```mermaid
graph LR
A(Your Door) -->B(Door Opened)
C(Location of Car) --> B
```

- berkson's paradox - diseases in hospitals are correlated even when they are not in the general population
  - possible explanation - only having both diseases together is strong enough to put you in the hospital
- simpson's paradox - see plot above where lines decrease given conditioning but increase overall



# reviews

- [Causality for Machine Learning](https://arxiv.org/abs/1911.10500) (scholkopf 19)
  - most of ml is built on the iid assumption and fails when it is violated (e.g. cow on a beach)

# interesting empirical studes

- Hainmueller & Hangartner (2013) - Swiss passport
  - naturalization decisions vary with immigrants' attributes
  - is there immigration against immigrants based on country of origin?
  - citizenship requires voting by municipality
- Sekhon et al. - when natural experiments are neither natural nor experiments
  - even when natural interventions are randomly as- signed, some of the treatment–control comparisons made available by natural experiments may not be valid
- Grossman et al. - "Descriptive Representation and Judicial Outcomes in Multiethnic Societies"
  - judicial outcomes of arabs depended on whether there was an Arab judge on the panel
- angrist_96_instruemtal
  - bridges the literature of instrumental variables in econometrics and the literature of causal inference in statistics
  - applied paper with delicate statistics
  - carefully discuss the assumptions
  - instrumental variables - regression w/ constant treatment effects
  - effect of veteran status on mortality, using lottery number as instrument
- [Sex Bias in Graduate Admissions: Data from Berkeley](https://homepage.stat.uiowa.edu/~mbognar/1030/Bickel-Berkeley.pdf) (bickel et al. (1975)
  - simpson's paradox example
- angrist_99_class_size
  - reducing class size induces a signi􏰜cant and substantial increase in test scores for fourth and 􏰜fth graders, although not for third graders.
- [cornfield_59_smoking](https://academic.oup.com/jnci/article/22/1/173/912572)
  - not a traditional statistics paper
  - most of it is a review of various scientific evidence about smoking and cancer
  - small methodology section that describes an early version of sensitivity analysis
  - describes one of the most important contributions causal inference has made to science
- rosenbaum_01_surgery
  - spends a lot of time dsiscussing links between quantitative and qualitative analyses
  - takes the process of checking assumptions very seriously, and it deals with an important scientific problem
- hansen_09_vote
  - about a randomized experiment
  - proved complex to analyze and led to some controversy in political science
  - resolves that controversy using well-chosen statistical tools.
  - Because randomization is present in the design I think the assumptions are much less of a stretch than in many settings (this is also the case in the Angrist, Imbens, Rubin paper)
- [Incremental causal effects](https://arxiv.org/abs/1907.13258) (rothenhausler & yu, 2019)

# using non-linear models

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
