# Learning "Healthy" Models in Machine Learning for Health 

**Marzyeh Ghassemi, UToronto**

- risk scores have been around in medicine for a while (but mostlly logistic type)
- data
  - learning from practice
    - don't want to model some doctors, who may be temporarily burnt out
  - randomized control trials are rare
    - inclusion/exclusion criteria are often very aggressive, making the results hard to generalize
    - over 10% of top journal studies are medical reversals (i.e. overturn previous practice)
  - EHR has become very dominant lately
    - usually secondary, making it hard to work with
    - heterogenous, sparse, and uncertain
- representations
  - want representations to separate underlying factors (e.g. HR, BP)
  - enable semi-supervised learning - factors explaining *P(X)* are useful for learning *P(Y|X)*
  - allow shared factors across many learning tasks
  - learn a good latent repr. using HHMs
- differential privacy - add some noise + clipping to gradients of DNN during training
  - makes it so no individual point is identifiable from DNN
  - can hurt performance on minority groups
- progress
  - what is done: early alert systems
  - next step: automatic tedious tasks
  - next step: imaging
  - long-term goal: how to reduce variance in care?
