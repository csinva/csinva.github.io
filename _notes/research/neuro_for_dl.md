---
layout: notes_without_title
section-type: notes
title: neuro for dl
category: research
---


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


<script type="text/bibliography">
@article{hubel1962receptive,
  title={Receptive fields, binocular interaction and functional architecture in the cat's visual cortex},
  author={Hubel, David H and Wiesel, Torsten N},
  journal={The Journal of physiology},
  volume={160},
  number={1},
  pages={106--154},
  year={1962},
  publisher={Wiley Online Library},
  url={http://onlinelibrary.wiley.com/wol1/doi/10.1113/jphysiol.1962.sp006837/abstract}
}

@article{singh2017consensus,
  title={A consensus layer V pyramidal neuron can sustain interpulse-interval coding},
  author={Singh, Chandan and Levy, William B},
  journal={PloS one},
  volume={12},
  number={7},
  pages={e0180839},
  year={2017},
  publisher={Public Library of Science},
  url={http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180839}
}

@article{herz2006modeling,
  title={Modeling single-neuron dynamics and computations: a balance of detail and abstraction},
  author={Herz, Andreas VM and Gollisch, Tim and Machens, Christian K and Jaeger, Dieter},
  journal={science},
  volume={314},
  number={5796},
  pages={80--85},
  year={2006},
  publisher={American Association for the Advancement of Science},
  url={http://science.sciencemag.org/content/314/5796/80.long}
}

@article{carandini2004amplification,
  title={Amplification of trial-to-trial response variability by neurons in visual cortex},
  author={Carandini, Matteo},
  journal={PLoS biology},
  volume={2},
  number={9},
  pages={e264},
  year={2004},
  publisher={Public Library of Science},
  url={http://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0020264}
}

@article{yamins2014performance,
  title={Performance-optimized hierarchical models predict neural responses in higher visual cortex},
  author={Yamins, Daniel LK and Hong, Ha and Cadieu, Charles F and Solomon, Ethan A and Seibert, Darren and DiCarlo, James J},
  journal={Proceedings of the National Academy of Sciences},
  volume={111},
  number={23},
  pages={8619--8624},
  year={2014},
  publisher={National Acad Sciences},
  url={http://www.pnas.org/content/111/23/8619}
}

@incollection{fukushima1982neocognitron,
  title={Neocognitron: A self-organizing neural network model for a mechanism of visual pattern recognition},
  author={Fukushima, Kunihiko and Miyake, Sei},
  booktitle={Competition and cooperation in neural nets},
  pages={267--285},
  year={1982},
  publisher={Springer}
}

@article{marr1976understanding,
  title={From understanding computation to understanding neural circuitry},
  author={Marr, David and Poggio, Tomaso},
  year={1976},
  url={https://dspace.mit.edu/handle/1721.1/5782}
}

@article{schuman2017survey,
  title={A survey of neuromorphic computing and neural networks in hardware},
  author={Schuman, Catherine D and Potok, Thomas E and Patton, Robert M and Birdwell, J Douglas and Dean, Mark E and Rose, Garrett S and Plank, James S},
  journal={arXiv preprint arXiv:1705.06963},
  year={2017},
  url={https://arxiv.org/abs/1705.06963}
}
</script