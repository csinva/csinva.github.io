# causality through invariance

**leon bottou, fair**

- ml 2 approaches
  - come up w/ heuristic
  - just learn from data
- we want to ignore spurious correlations
- shuffling the data is a loss of info
  - assume data comes from many environments (we only see a few)
  - goal: do well on all of them
- robust approach: minimize max error on any environment - ends up mixing the environments w/ correct proportions
- ex. frames of movies / images have different corrs - sometimes taking call without phone
- invariant regression - simultaneously minimize the error in each environment
  - ex. noiseless - we learn exactly the right function
  - otherwise, must design model which is insensitive to the spurious correlation (want repre $\phi(X)$ which is invariant accross environments
- related work
  1. invariance and causation - e.g. Pearl's do-calculus on graphs or Rubin's ignorability assumption
  2. invariant causal prediction (peters, buhlman, meinshausen 16) - environments result from interventions on a causal graph
  3. adversarial domain adaption - want to learn classifier where it is hard to recover $e$ from $\phi(x)$
  4. robust supervised learning - want invariance to other training environment distrs.
- **the linear case**
  - solns are on ellipsoids
  - some interesting math...
  - has double-descent like phenomenena....
- invariant regularization
  - given two types of $e$ (e.g. color in 0.7, 0.8), force repr to be the same
    - when color now shows up 0.1 times, still works
  - color mnist is non-realizable - having more data would not fix the problem
- realizable case - if you have enough data, should learn the right function
  - invariance buys extrapolation powers - challenge is not to find right func but to find it faster