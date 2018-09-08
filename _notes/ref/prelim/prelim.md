# general

- linear algebra
  - pca
- info theory

# ai

- search
  - uninformed
  - a*
  - csps
  - local search
- decisions
  - game trees
  - logic
  - utilities
  - decision theory
  - vpi
  - mdps
    - **pomdps**
  - rl
- graphical models
  - independence / factorization
  - elimination
  - propagation-factor graphs
  - hmms / kalman filters

# ml

- optimization
  - em
- statistical estimation
  - multivariate gaussians
- generative
  - density estimation models
  - naive bayes
- discriminative
  - linear reg.
  - logistic reg.
  - svms / kernel methods
  - decision trees
  - nearest neighbor
  - neural nets (usually)

- misc facts
  - bayes estimator minimizes **risk** (expected loss, varies with loss func)
  - hard-margin + soft differ for separable data (c=inf when same)
  - can't use implies with there exists (trivially true)
  - bayes net can't have loops
  - left-singular is left vector in SVD
  - averaging multiple trees (decreases var, doesn't increase)
  - for alpha-beta pruning, usually pick correct soln first
  - uniform distr can be represented by any bayes net
  - likelihood weighting only takes upstream evidence into account when sampling (unlike Gibbs)

# todo

- memorize new cheat sheet (logic, planning)
- practice problems
  - 188 finals / discussions
  - 189 finals / discussions
  - russell qs
  - **do exam-prep** - 7 q2
- review topics
  - approximate q-learning?
    - rl eqs
  - conditional independencies
    - forward-backward algo
  - ac-3 / backtracking
  - kernels
    - svms duality
