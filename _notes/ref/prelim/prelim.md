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

# misc facts

- bayes estimator minimizes **risk** (expected loss, varies with loss func)
- can't use implies with there exists (trivially true)
- left-singular is left vector in SVD
- averaging multiple trees (decreases var, doesn't increase bias)
- for alpha-beta pruning, usually pick correct soln first
- kd tree: binary split on different dim each time
- graphical models
  - uniform distr can be represented by any bayes net
  - bayes net can't have loops
  - likelihood weighting only takes upstream evidence into account when sampling (unlike Gibbs)
  - conditional independencies - remember bayes ball algo
    - bounce through all non-shaded unless shaded is base of v
    - can bounce up farther from base of v
  - gibb's sampling: need to resample (using proportionality) keeping all other vars constant
  - state space keeps track of things that change
  - filtering, prediction, smoothing, mle
    - viterbi: $m_t[x_t] = P (e_t|x_t) \max P (x_t|x_t−1)m_{t−1}[x_{t−1}]$
- qda/svm with quadratic kernel can represent hyperbola boundary
- As the number of neighbors k gets very large and the number of training points goes to infinity, the probability of error for a k-nearest-neighbor classifier will tend to approach the probability of error for a MAP classifier that knew the true underlying distributions. 
- svm
  - soft margin SVM effectively optimizes hinge loss plus ridge regularization
  - If a training point has a positive slack, then it is not a support vector.
  - hard-margin + soft differ for separable data (c=inf when same)
- remember to justify convexity!
- $x\sim N(\mu, \Sigma) \implies v^Tx ~ N(v^T\mu, v^T\Sigma v)$
- don't write $\sqrt{\Sigma}$ write $\Sigma^{1/2}$
- logistic reg: $P(Y=y|x, \theta) = p(y)^{y} \cdot (1-p(y))^{y}$
- csp
  - Any node can be backtracked on up until a cutset has been assigned. Note that B’s values in the first part has no effect on the rest of the CSP after A has been assigned. However, because of the way that backtracking search is run, B would still be re-assigned before A if there was no consistent solution for a given value of A
  - tree-structured csp algo (has backward pass for arc consistency / forward pass for assigning variables - faster)
  - when enforcing arc consistency before every value assignment, we will only be guaranteed that we won’t need to backtrack when our remaining variables left to be assign form a tree structure 
- $MEU(B) = P(B=0) MEU(B=0) + P(B=1)MEU(B=1)$
  - note we calculate MEU with respect to an assignment of a variable