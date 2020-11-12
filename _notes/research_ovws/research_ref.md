[TOC]

# data

- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html#sklearn.datasets.fetch_openml
  - sklearn also lets you generate some nice synthetic datasets
- pmlb

# misc

- anonymous github: [https://anonymous.4open.science](https://anonymous.4open.science/)
- posters
  - new [simple poster template](https://www.youtube.com/watch?v=1RwJbhkCA58&feature=youtu.be)
  - infographics don't work great
  - "perfection is not when you have nothing to add, it's when you have nothing to take away"
- [draw arch](http://alexlenail.me/NN-SVG/index.html)
- [viz architectures](https://tensorspace.org/index.html)
- medical ideas: many diseases manifest themselves in the activity of neurons, not in the structure
- ovw![ai_generations](../assets/ai_generations.png)

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

# small misc things

- neuroscience is like IT - given computer, figure out how it works
- https://grand-challenge.org/all_challenges/

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

### brain on a chip

- neuromorphic chips
- grow cells in vivo

### connectomics

- C Elegans
  - 302 neurons
  - no evidence of Hebbian learning
  - develop synaptogenesis rules?

### history

- biology U: phenomenon (high level) -> element (low level) -> synthesis (high level)


### cogsci

- model-free vs model-based



### biomechanics

- brain exists to make suggestions to motor system
- reflexes don't need cortex, but still tricky
  - ex. wipe skin when irritated in frog - works when leg at different points, leg stopped
- idea: t-sne on neural dynamics: someone at Emory
- spinal chord is where reflexes are, then brainstem, and cortex is quite slow


# scientists

- ml
  - *Geoffrey Hinton* - emeritus, Toronto
    - *Yann Lecun* (NYU) - heads fb AI
      - was his research associate
  - *Michael Jordan* - Berkeley
    - students: Ng, Blei; postdoc: Bengio
    - *Andrew Ng* - Stanford
    - *David Blei* - topic modeling
    - *Yoshua Bengio* - McGill
      - *Ian Goodfellow* - GANs
  - *Jeff Hawkins* - established redwood
    - left to found numenta
  - *Terry Sejnowski*
    - coinvented Boltzmann machines
  - *Daphne Koller* - co-founder of Coursera
    - representation, inference, learning, and decision making
  - *Schmidhuber & Hochreiter* - LSTM
  - *Jitendra Malik* - computer vision
  - *Andrej Karpathy* - blogger, Tesla AI director
- comp neuro
  - *Karl Friston* - functional imaging analysis
  - *Raymond Dolan* - emotion, pain
  - *Terrence J. Sejnowski* - boltzmann machines, ICA
  - *david marr* - vision
  - *tomaso poggio* - vision
  - *Christoph Koch* - head of allen institute
  - *Daniel Wolpert* - noise in the nervous system
  - *Jonathan Cohen* - theory
  - *Larry Abbott* - theoretical neuroscience
  - *György Buzsáki* - oscillations
  - Peter Dayan
  - Haim Sompolinsky 
  - Stephen Grossberg
  - Randall C. O'Reilly
  - Nancy Kopell
  - Chris Eliasmith
  - Michael Hasselmo
  - David Heeger
  - Roger D. Traub
  - Bard Ermentrout
  - Eugene M. Izhikevich
  - Eric L. Schwartz
- misc
  - *Byron Yu* - bmi
  - *James DiCarlo*
  - *Liam Paninski* - decoding
  - *Jack Gallant* - v4, fmri
  - *Bin Yu* - model consistency
  - *Sebastian Seung* - connectomes
  - *Surya Ganguli* - Stanford
  - *David Cox* - MIT/IBM
  - *Jascha Sohl-Dickstein* - Google
  - [Iain Couzin](https://scholar.google.com/citations?user=dbBW62EAAAAJ&hl=en)
  - Haim Sompolinksy
  - charlest gilbert - rockefeller - studies spatial distribution of visual cortex
  - *Susumu Tonegawa* - memory