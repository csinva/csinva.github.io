---
layout: notes_without_title
section-type: notes
title: causal inference
category: research
---

[TOC]

Some interesting / seminal papers in causal inference along with my notes.

# interesting empirical studes

- Hainmueller & Hangartner (2013) - Swiss passport
  - naturalization decisions vary with immigrants' attributes
  - is there immigration against immigrants based on country of origin?
  - citizenship requires voting by municipality
- Sekhon et al. - when natural experiments are neither natural nor experiments
  - even when natural interventions are randomly as- signed, some of the treatment–control comparisons made available by natural experiments may not be valid
- Grossman et al. - "Descriptive Representation and Judicial Outcomes in Multiethnic Societies"
  - judicial outcomes of arabs depended on whether there was an Arab judge on the panel

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

# classical papers

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
  - Because randomization is present in the design I think the assumptions are much less of a stretch than in many settings (this is also the case in the Angrist, Imbens, Rubin paper).