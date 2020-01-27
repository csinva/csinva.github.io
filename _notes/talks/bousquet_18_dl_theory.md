# what would dl theory look like?

# observations

- zhang et al. paper - dl has 0 train error
- test err = generalization gap
- over-parameterization helps optimization
- not clear what is minimized (which local min is picked) - no obvious regularization

## other examples

- kNN
  - 0 training err
  - generalize if there is structure, otherwise overfits
  - but
    - regularization understood through use of k
    - non-parametric
    - no "training"
- SVM
  - large capacity
  - non-parametric
- boosting
  - training err keeps going down, test goes down too

## questions

- consistency?
- convergence rates for a fixed "regularization"
- generalization

## curious

- dl exhibits properties of margin maximization - implicit regularization

## regularization possibilities

- initialization
- discreteness of GD
- noise of SGD
- other noise (e.g. dropout

## optimization

- flat minima - different regularization try to go towards or away from flat minima
  - unclear if flat minima are betterchro
