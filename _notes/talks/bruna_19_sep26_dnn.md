# On Large Deviations for Large Neural Networks

**joan bruna, nyu**



- assume noise-free setting, mse loss
- limit "complexity": ||f|| < $\delta$
- ben recht's "fundamental thm of ml"
  - three terms: approximation error, generalization error, optimization error
- classic functional spaces don't work in high dims
  - ex. lipschitz doesn't work
  - ex. sobolev doesn't work
  - ex. dnns seem to work
- simplifying assumption: let step size go to zero and look at continuous-time gradient flow
  - single hidden layer - vary num inputs $L$, neurons $n$, input dim $d$
  - mean field limit - describe w/ pde
  - scaling is consisten w/ **variation norm spaces**
- understanding
  - can understand it w/ a reproducing kernel hilbert space (rahimi/recht 08) - however in high dim requires functions to be very smooth (bach' 16)
  - total variation spaces: instead, look at Banach space, limiting the TV norm
    - his includes sums of ridge functions (e.g. 1 hidden layer)
  - consider parameters particles and see what happens when number of particles grows
    - mean field thm: time evolution of system can be studied by studying specific initialization
  - thm: rate of global convergence - avoid local fixed points of pde
    - generalization err also
- proposal: non-local modification dynamics based on unbalanced transport using birth/death processes - improve guarantees