---
layout: notes
section-type: notes_without_title
title: neural coding
category: blog
---


# neural coding
**chandan singh** 
*last updated jul 20, 2018*

---


Biological neurons are incredibly complex. Their computations involve integrating inputs from thousands of other neurons over time and space, with sophisticated molecular machinery. The research here explores these computations in detail through biophysical simulations. Surprisingly, simplicity often emerges from all of this complexity.

# Linearization of synaptic integration

*"Linearization of excitatory synaptic integration at no extra cost", <a color="#219AB3" href="http://rdcu.be/FDUo"> Morel, Singh, & Levy, 2018, Journal of Computational Neuroscience</a>*

The basic question here is: "Can a complex, realistic neuron model behave like a simple perceptron and simply sum its inputs?" The results suggest they can: for an increase in the synaptic input x, a realistic model can yield a linear increase in somatic voltage c*x. This simplicity is surprising as we know neurons have complex integration processes, but simplicity seems to emerge from all the complexity.

Furthermore, the results show that it requires no extra energy for a neuron model to linearly sum its inputs. This is also surprising, since synaptic excitation tends to decay as signals travel from the synapses to the soma, and intuitively it seems that neurons would require expending extra energy to counteract this.

# Interpulse interval coding

*"A consensus layer V pyramidal neuron can sustain interpulse-interval coding" -  <a color="#219AB3" href="http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180839"> Singh & Levy, 2017, Plos One</a>.*

## Coding
Neurons communicate via spikes called action potentials. It is unclear exactly how these spikes transmit information. There are two general hypotheses: 
​    
1. rate codes - neurons communicate with the rate of their firing, so firing more would send a higher signal 
2. spike-timing codes (or interpulse-interval codes) - neurons communicate the precise timing between their spikes

Spike timing codes are more precise than rate codes, but it's unclear if real neurons can sustain interpulse interval coding because of how much noise there is in their processing. This study quantifies effects of this noise and finds that certain neurons, can in fact sustain interpulse interval coding.

## Summary
In order to obtain a complete understanding of neural computation and communication, it is necessary to understand how one neuron communicates its computation with another. To create a quantitative understanding of such communication, we need to understand the neural code. One possible code is referred to generically as a spike-timing code, or in a more technical way, an interpulse interval code. Here, we study this code using a biophysical model of a neocortical pyramidal neuron consisting of an appropriate morphology and known voltage-activated channels. We consider this neuron's time-to-spike as an appropriate encoding of the hidden intensity which controls the neuron's inputs. We find that this biologically appropriate and biophysically complex neuron has a simple characterization for interpulse interval coding, a characterization not demonstrated by the biophysically simpler passive model often used by neurotheoreticians or by more complicated neurons specified through two or three differential equations. Thus, we have an example where biophysical complexity, at the level of cell morphology and voltage-activated channels, leads to input-output simplicity at the level of a neuronâs encoding of information.

# Backpropagating action potentials

*"Examining the Effect of Action Potential Back Propagation on Spike Shape" - <a color="#219AB3" href="/assets/singh_14_dendrite_backprop.pdf"> Singh 2014, Unpublished</a>.*

# Action potential velocity