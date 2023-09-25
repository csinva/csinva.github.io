---
layout: notes
title: running notes on evaluating interpretability
category: blog
---

- Interpretability is gaining increased attention, but also [coming under much criticism](https://arxiv.org/abs/1606.03490)
- The purpose of interpretations is to [help a particular audience solve a particular task](https://www.pnas.org/content/116/44/22071). It is important that the evaluations reflect this.
- It is unclear if modern methods are getting better at downstream tasks: **better-looking explanations $\neq$ better explanations.**
    - this is because they are [often evaluated qualitatively](https://arxiv.org/abs/1810.03292), accumulating human biases.
    - Some methods [provably find nice-looking interpretations](https://arxiv.org/abs/1805.07039), such as edges in an image, not being faithful to the model.
    - Moreover, with the barrage of new methods, it is difficult to know which method to pick for a problem
    - Interpretations are also [fragile](https://arxiv.org/abs/1710.10547)
- As of now, it is unclear even what forms these methods should take resulting in many different alternatives: e.g. [saliency maps](https://arxiv.org/abs/1705.07857), [concept activation vectors](https://arxiv.org/abs/1711.11279), [hierarchical feature importances](https://arxiv.org/abs/1806.05337), importance curves, and [textual explanations](https://arxiv.org/abs/1411.4389). Proper metrics can help guide the development of new forms of intepretability.
- We want to introduce a set of downstream tasks which serve as baselines for whether interpretability methods are useful and drive the innovation of new forms of useful interpretation. They can be used to derive desiderata for new methods or to evaluate their performance. It is very difficult for the field to move forward until it knows what it is trying to move towards.
- Note: whenever possible, model-based interpretability (i.e. using an easily understandable model) is [preferable to post-hoc interpretability](https://arxiv.org/abs/1811.10154) (i.e. interpreting a trained black-box model). Claims that interpretability will work for things like medicine and self-driving cars are errant unless they can reliably work.

# work on evaluating interpretability

- [Doshi-Velez and Kim](https://arxiv.org/pdf/1702.08608.pdf) break apart evaluation interpretability into three types
    1. application-grounded (real humans doing real tasks), 
    2. human-grounded evaluation (real humans, simple tasks)
    3. functionally-grounded evaluation (no real humans, proxy tasks)
- [another work](https://arxiv.org/pdf/1902.00006.pdf) proposes 3 cognitive tasks:
    1. **Simulation** - Predicting the system’s recommendation given an explanation and a set of input observations.
    2. **Verification** - Verifying whether the system’s recommendation is consistent given an explanation and a set of input observations.
    3. **Counterfactual** -  Determining whether the system’s recommendation changes given an explanation, a set of input observations, and a perturbation that changes one dimension of the input observations.
- [Predictive vs descriptive accuracy](https://www.pnas.org/content/116/44/22071). We continue to use the framework of predictive vs descriptive, where these are often in conflict. Here, the focus is on measuring descriptive accuracy.
- Some have proposed metrics to evaluate specific methods. For example, one recent paper proposes a possible method for [evaluating feature importance estimates](https://arxiv.org/abs/1806.10758). Similarly, [RISE](https://arxiv.org/abs/1806.07421) uses random masks on insertion/deletion. These are useful, but still don't solve the overall problem of not knowing the correct form and it is unclear what downstream tasks these evaluations are useful for.

# concrete tasks

1. improving performance such as predictive accuracy or sample efficiency (finding scores that generalize)
    - for imagenet, this can take the form of refining a model
    - finding failure modes of a model and automatically flipping them works
    - feature engineering: extracting interactions and adding them to a linear model (e.g. on simple datasets - pmlb)
    - show that debugging helps find errors
    - has 2 parts: (1) get the explanation (2) how to use it?
    - might improve data efficiency by having a human in the loop
    - sanity-checks on things like preprocessing, that model is looking at correct features etc.
2. predicting uncertainty/robustness (can I trust this prediction?)
    - should I trust this prediction or not?
    - does uncertainty correlate to "groundtruth" uncertainty - probability a model would actually get something wrong?
    - can we use this to identify when a model will fail?
    - e.g. Bayesian neural nets give explicit uncertainties
    - e.g. [influence funcs paper](https://arxiv.org/abs/1703.04730) flips labels and then finds them
3. discovering/fixing models (things like bias)
    - often we use interpretability to "fix" something about a model
    - want to make the fix generalize (i.e. we find bias is gone inside the model, then we want to make sure this fix works on new data)
    - [right for the right reasons](https://arxiv.org/abs/1909.13584)
4. finding causal descriptions (in the data)
    - causal inference finds causal relationships in the data (and has a long history of using simulations)
    - finding causal relationships about the model is different, and can be much simpler
        - descriptions can help summarize something for fundamental understanding (e.g. science)
        - feature importances are often trying to get at some notion like this
    - discovering interactions
    - object deletion seems like a decent task, since it preserves structure of the scene
    - [could use gan to fill in missing region](https://arxiv.org/abs/1807.08024)


# descriptive accuracy

All the above tasks implicitly require that the model has high descriptive accuracy (particularly finding causal patterns in data). This type of descriptive accuracy can be measured via simulation.

- for images, this requires a new simulation: driving simulator? 3d models projected to 2D? Images of cats w/ eye colors?
 - need to evaluate via simulations
    - ex. finding groundtruth via [simulations of images](https://arxiv.org/abs/1712.06302), alter dataset (e.g. texture bias turns into a shape bias by [removing texture](https://arxiv.org/abs/1811.12231) 
        - one problem with this is the model could be mislead and accidentally attribute importance to the wrong part of the image
        - another good baseline is removing objects from images
    - for text, this evaluation would entail evaluating faithfulness to the **model**, not the input (we shouldn't be able to change the model's prediction without changing the explanation and vice versa)


# examples

- we are constantly striving to evaluate interp and make it better (e.g. human experiments in the [ACD](https://openreview.net/pdf?id=SkEqro0ctQ) paper, simulation experiments in the [TRIM](https://arxiv.org/abs/2003.01926)/[DAC](https://arxiv.org/abs/1905.07631) papers, improving predictive accuracy in [CDEP](https://arxiv.org/abs/1909.13584) paper)