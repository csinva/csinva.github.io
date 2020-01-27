# building probabilistic structure into masssively parameterized models - ryan adams

- semiparametric view of the world
- if you give this mouse a drug / modify its genes / stimulate its brain -> how does its behavior change?
  - ex. 1 number
- learn a language of behavior (from videos)
  - ex. dart, pause, rear
- model
  - discrete hidden state (with markov transitions)
  - induces continuous observed state
- deep learning prior (convolutional invariance) is really hard to posit as a graphical model
- require complex data + structured model (semiparametric)
- neural net helps to make observed continuous really complicated
- variational autoencoder
  - encoder: data $\to$ vector (posterior approx)
  - decoder: vector $\to$ data (likelihood)
- use neural network to account for model mismatch (ex. data is not really Gaussian)
- data given frames would like to say each frame is distinct
  - need to introduce bias to overcome this
- also get markov transition matrix between states