{:toc}

# Ideas for deep learning from neuroscience

This aims to be a primer on aspects of neuroscience which could be relevant to deep learning researchers. These two communities are becoming more intertwined, and could benefit greatly from each other. However, current literature in neuroscience has a steep learning curve, requiring learning much about biology. This primer aims to equip deep learning researchers with the basic computational principles of the brain, to draw inspiration and provide a new perspective on neural computation.

## Explaining concepts from neuroscience to inform deep learning

Modern deep learning evokes many parallels with the human brain. Here, we explore how these two concepts are related and how neuroscience can inform deep learning going forward <dt-fn>Note that this post largely ignores the important reverse question: how can deep learning inform neuroscience?</dt-fn>

The brain currently outperforms deep learning in a number of different ways: efficiency, parallel computation, not forgetting, robustness. Thus, in these areas and others, the brain can offer high-level inspiration as well as more detailed algorithmic ideas on how to solve complex problems.

We begin with some history and perspective before further exploring these concepts at 3 levels: (1) the neuron level, (2) the network level, and (3) high-level concepts.

## Brief history
The history of deep learning is intimately linked with neuroscience. In vision, the idea of hierarchical processing dates back to Hubel and Weisel <dt-cite key="hubel1962receptive"></dt-cite> and the modern idea of convolutional neural networks dates back to the necognitron<dt-cite key="fukushima1982neocognitron"></dt-cite>.

Ranges from neurally-inspired -> biologically plausible

Computational neuroscientists often discuss understanding computation at Marr's 3 levels of understanding: (1) computational, (2) algorithmic, and (3) mechanistic<dt-cite key="marr1976understanding"></dt-cite>. The first two levels are most crucial to understanding here, while the third may yield insights for the field of neuromorphic computing <dt-cite key="schuman2017survey"></dt-cite>.

## Cautionary notes

There are dangers in deep learning researchers constraining themselves to biologically plausible algorithms. First, the underlying hardware of the brain and modern von Neumman-based architectures is drastically different and one should not assume that the same algorithms will work on both systems. Several examples, such as backpropagation, were derived by deviating from the mindset of mimicking biology.

Second, the brain does not solve probleDangers for going too far.... One wouldn't want to draw inspiration from the retina to put a hole in the camera.


<img width="50%" src="figs/retina.png"></img>
Gallery of brain failures. Example, inside-out retina, V1 at back...


# Neuron level

The fundamental unit of the brain is the neuron, which takes inputs from other neurons and then provides an output.

Individual neurons perform varying computations. Some neurons have been show to linearly sum their inputs <dt-cite key="singh2017consensus"></dt-cite>

 - neurons are complicated (perceptron -> ... -> detailed comparmental model)

For more information, see a very good review on modeling individual neurons<dt-cite key="herz2006modeling"></dt-cite>.

 - converting to spikes introduces noise <dt-cite key="carandini2004amplification"></dt-cite>- perhaps just price of long-distance communication

# Network level

 Artificial neural networks can compute in several different ways. There is some evidence in the visual system that neurons in higher layers of visual areas can, to some extent, be predicted linearly by higher layers of deep networks<dt-cite key="yamins2014performance"></dt-cite>. However, this certainly isn't true in general. Key factors

 For the simplest intuition, here we provide an example of a canonical circuit for computing the maximum of a number of elements: the winner-take-all circuit.

 Other network structures, such as that of the hippocampus are surely useful as well.

 Questions at this level bear on population coding, or how groups of neurons jointly represent information.

# High-level concepts

 Key concepts differentiate the learning process. Online,
- learning

- high-level
  - attention
  - memory
  - robustness
  - recurrence
  - topology
  - glial cells

- inspirations
  - canonical cortical microcircuits
  - nested loop architectures
  - avoiding catostrophic forgetting through synaptic complexity
  - learning asymmetric recurrent generative models
- spiking networks ([bindsnet](https://github.com/Hananel-Hazan/bindsnet))
- neural priors
  - cox...

# Conclusion

A convergence of ideas from neuroscience and deep learning can be useful.