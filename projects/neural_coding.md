---
layout: notes
title: Neural coding
---

Biological neurons are incredibly complex. Their computations involve integrating inputs from thousands of other neurons over time and space, with sophisticated molecular machinery. The research here explores these computations in detail through biophysical simulations. Surprisingly, simplicity often emerges from all of this complexity.

# Linearization of synaptic integration

*"Linearization of excitatory synaptic integration at no extra cost", <a color="#219AB3" href="http://rdcu.be/FDUo"> Morel, Singh, & Levy, 2018, Journal of Computational Neuroscience</a>*

The basic question here is: "Can a complex, realistic neuron model behave like a simple perceptron and simply sum its inputs?" The results suggest they can: for an increase in the synaptic input x, a realistic model can yield a linear increase in somatic voltage c*x. This simplicity is surprising as we know neurons have complex integration processes, but simplicity seems to emerge from all the complexity.

Furthermore, the results show that it requires no extra energy for a neuron model to linearly sum its inputs. This is also surprising, since synaptic excitation tends to decay as signals travel from the synapses to the soma, and intuitively it seems that neurons would require expending extra energy to counteract this.

# Interpulse interval coding

"A consensus layer V pyramidal neuron can sustain interpulse-interval coding" -  <a color="#219AB3" href="http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180839"> Singh & Levy, 2017, Plos One</a>.

# Backpropagating action potentials

"Examining the Effect of Action Potential Back Propagation on Spike Shape" - <a color="#219AB3" href="/assets/singh_14_dendrite_backprop.pdf"> Singh 2014, Unpublished</a>.