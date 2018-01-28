---
layout: notes
section-type: notes
title: research ref
category: research
---

[TOC]

# datasets

- keeping up to date: https://sanjayankur31.github.io/planet-neuroscience/
- *lots of good data*: http://home.earthlink.net/~perlewitz/index.html
- *models*
  - senseLab: https://senselab.med.yale.edu/
    - modelDB - has NEURON code
  - model databases: http://www.cnsorg.org/model-database 
  - comp neuro databases: http://home.earthlink.net/~perlewitz/database.html
- *raw misc data*
  - crcns data: http://crcns.org/
    - visual cortex data (gallant)
    - hippocampus spike trains
  - allen brain atlas: http://www.brain-map.org/
    - includes calcium-imaging dataset: http://help.brain-map.org/display/observatory/Data+-+Visual+Coding
  - wikipedia page: https://en.wikipedia.org/wiki/List_of_neuroscience_databases
- *human fMRI datasets*: https://docs.google.com/document/d/1bRqfcJOV7U4f-aa3h8yPBjYQoLXYLLgeY6_af_N2CTM/edit
  - Kay et al 2008 has data on responses to images
- *calcium imaging* for spike sorting: http://spikefinder.codeneuro.org/
  - spikes: http://www2.le.ac.uk/departments/engineering/research/bioengineering/neuroengineering-lab/software

# levels

- *Marr's three levels*
  1. computation - behavior
  2. algorithm - causal connectomics
  3. basic implementation - structural connectmoics, molecular biology

# questions

- *problems that are solved, or soon will be*
  - how do single neurons compute?
  - what is the connectome of a small nervous system, like that of C. elegans (300 neurons)?
  - how can we image a live brain of 100,000 neurons at cellular and millisecond resolution?
    - hydra was completed
  - how does sensory transduction work?
- *problems that we should be able to solve in the next 50 years*
  - can we add senses to the brain?
    - like cochlear implant
    - like vibrations
  - how do circuits of neurons compute?
  - what is the complete connectome of the mouse brain (70e6 neurons)?
  - how can we image a live mouse brain at cellular and millisecond resolution?
  - what causes psychiatric and neurological illness?
  - how do learning and memory work?
    - short-term vs. long-term
    - declarative vs. non-declarative
    - encodes relationships between things not things themselves
    - memory retrieval
  - why do we sleep and dream?
    1. sleep is restorative (but then why high neural activity?)
    2. allows the brain to run simulations
    3. consolidating memories and forgetting
  - where is consciousness?
    - at this point, sounds and vision should line up (delayed appropriately)
  - how do we make decisions?
  - how does the brain represent abstract ideas?
  - what does neural baseline activity represent?
  - how does the brain solve timing?
    - moving eyes
    - blinking
    - hearing and vision time differences
  - how does sensorimotor learning build a model of the world?
- *problems that we should be able to solve, but who knows when*
  - how do brains simulate the future?
  - how does the mouse brain compute?
  - what is the complete connectome of the human brain (8e10 neurons)?
  - how can we image a live human brain at cellular and millisecond resolution?
  - how could we cure psychiatric and neurological diseases?
  - how could we make everybody’s brain function best?
  - brain and quantum?
    - some work in quantum neural nets
  - how is info coded in neural activity?
    - like measuring tansistors and guessing what computer is doing
    - neuron gets lots of inputs
  - do glial cells and other signaling molecules compute?
  - what is intelligence?
    - what is iq?
  - how do specialized systems integrate?
- *problems we may never solve*
  - what are emotions?
    - brain states that quickly assign values
    - in the amygdala
  - how does the human brain compute?
  - how can cognition be so flexible and generative?
  - how and why does conscious experience arise?
    - thing that flickers on when you wake up that was not there
    - evolutionary to manage all the different systems
- *meta-questions*
  - what counts as an explanation of how the brain works? (and which disciplines would be needed to provide it?)
  - how could we build a brain? (how do evolution and development do it?)
  - what are the different ways of understanding the brain? (what is function, algorithm, implementation?)
- ref David Eaglemen article: http://discovermagazine.com/2007/aug/unsolved-brain-mysteries
- ref Adolphs 2015, "The unsolved problems of neuroscience"

# problems

### rna barcoding
- allows for tagging different neurons
  - can then optically get differences
  - also can sequence and get differences (http://www.cell.com/neuron/pdf/S0896-6273%2816%2930421-4.pdf)
- future of electrophysiology: https://www.technologynetworks.com/neuroscience/articles/shining-a-light-on-the-future-of-electrophysiology-286992

### brain transplant

- computational hypothesis of the mind

### tms
- temporary cure for autism
- can change people's minds

### quantum brain

- quantum brain?

### connectomics

- C Elegans
  - 302 neurons
  - no evidence of Hebbian learning
  - develop synaptogenesis rules?

### history

- biology U: phenomenon (high level) -> element (low level) -> synthesis (high level)



# misc

- 4 axioms of neurobiology
  - neuron doctrine - neuron is basic unit
  - different stereotyped cell types
  - action potentials flow in -> out axon
  - canonical circuits
- what is the min descreption we need to describe a neuron?
  - depends on which neuron, which scenario
  - purkinje neurons may just be linear - they look like trees
- neurons aren't point neurons
  - dendrites - need multicompartments
  - NMDA "spikes" - local
  - summation between branches are more nonlinear than on the same branch
- Hubel 82: simple cells could be skipped and replaced by dendrites
- dendritic subunits are not locationless
  - distal and proximal inputs combine in complex ways
  - ![](assets/neuron_models.JPG)
- spike firing lights up entire basal dendrites
  - local dendritic computation can still maintain individuality despite APs
  - basal dendrites still linear despite APs
  - charging is slow - ignores spikes
    - spikes help to linearize
    - locks down IV curve of basal synapses
  - backpropagating spike can be learning signal
- encoding info w/ spikes
  - rate
  - firint time wrt ref signal (ex. oscillation)
  - simultaineuos firing of several neurons
  - relative timing of spikes
- synapses only work ~50% of the time - ***like dropout***
  - Branco...Nat rev. neurosci 2009
  - kavalali...Nat rev neurosci 2015 
- compare neuron with Boltzmann machines
  - bin times each neuron 0 or 1 if firing during the bin
  - Boltzmann machine each time step 0 or 1
  - n-bit vector dynamics
- some evidence for finding solns through fast sampling
- ideas
  - diversity of neurons + synapses, synapse has complex temporal dynamics
  - generic firing patterns
  - hard to easily make BNN that can do subtraction
    - can't transmit negative numbers
  - dynamical system (firing rate vector) evolves over time
- boyd + chua 1985 - diversity of units may increase the computational capability of a network
- liquid computing model
- neural codes are highly correlated - doesn't fit w/ theory
- synatpic plasticity + neural codes
  - STDP / hebbian learning seem to work sometimes
  - rewiring happens even w/out firing activity
  - neural codes drift
  - anything that remains invariant?
- excite + inhibit at same time lowers gain - future EPSP will be smaller
- monkey v2 latency is less than v1!
- corticocortical connections tend to have parallel connection that goes through thalamus
  - fastest axons are cotex -> thalamus -> cortex
  - connections through thalamus have almost same latency regardless of start/end area
- let's not impose artificial constraints based on cartoon models of topics in science that we don't yet understand.

### biomechanics

- brain exists to make suggestions to motor system
- reflexes don't need cortex, but still tricky
  - ex. wipe skin when irritated in frog - works when leg at different points, leg stopped
- idea: t-sne on neural dynamics: someone at Emory

### data types


|              | EEG      | ECoG              | Local Field potential (LFP) -> microelectrode array | single-unit | calcium imaging | fMRI     |
| ------------ | -------- | ----------------- | ---------------------------------------- | ----------- | --------------- | -------- |
| scale        | high     | high              | low                                      | tiny        | low             | high     |
| spatial res  | very low | low               | mid-low                                  | x           | low             | mid-low  |
| temporal res | mid-high | high              | high                                     | super high  | high            | very low |
| invasiveness | non      | yes (under skull) | very                                     | very        | non             | non      |

- newer
  - neural dust
  - neuropixels

# ongoing projects
- gov-sponsored
  - human brain project
    - blue brain project- large-scale brain simulation
  - european brain project
- companies
  - Neuralink
    - surgeons won't want to put chips into people's brains: http://waitbutwhy.com/2017/04/neuralink.html
  - Kernel
  - Facebook neural typing interface
  - google brain / google deepmind
  - IBM: project joshua blue

#conferences

|         | 2018                   | 2019              |
| ------- | ---------------------- | ----------------- |
| AAAI    | 2/2 - New Orleans      |                   |
| COSYNE  | **3/1 - Denver**       |                   |
| AISTATS | 4/9 - Playa            |                   |
| VSS     | 5/18 - Florida         | Florida           |
| SAND    | ~5/31 - Pitt           | ~5/31 - Pitt      |
| AREADNE | 6/20 - Greece          |                   |
| CVPR    | 6/18 - Salt Lake City  |                   |
| CNS     | ~July - Seattle        |                   |
| ICML    | ***7/10 - Stockholm*** | 7/10 - Long Beach |
| ECCV    | 8/8 - Munich           |                   |
| CCN     | 9/6                    |                   |
| ICCV    | X                      |                   |
| SFN     | 11/3 - San Diego       | 10/19 - Chicago   |
| NIPS    | 12/3 - Montreal        |                   |
|         |                        |                   |
# areas

- Basic approaches
  -  The problem of neural coding
  -  Spike trains, point processes, and firing rate
  -  Statistical thinking in neuroscience
  -  Overview of stimulus-response function models
  -  Theory of model fitting / regularization / hypothesis testing
  -  Bayesian methods
  -  Estimation of stimulus-response functionals:  regression methods, spike-triggered covariance
  -  Variance analysis of neural response
  -  Estimation of SNR. Coherence
  -  Generalized Linear Models
- Information theoretic approaches:
  -  Information transmission rates and maximally informative dimensions
  -  Scene statistics approaches and neural modeling
- Techniques for analyzing multiple-unit recordings:
  - Event sorting in electrophysiology and optical imaging
  - Optophysiology cell detection
  - Sparse coding/ICA methods, vanilla and methods including statistical models of nonlinear dependencies
  - Methods for assessing functional connectivity
  - Statistical issues in network identification
  - Low-dimensional latent dynamical structure in network activity–Gaussian process factor analysis/newer methods
- Models of memory, motor control and decision making:
  - Neural integrators
  - Attractor networks
