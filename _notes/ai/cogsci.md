---
layout: notes
title: Cognitive science
category: ai
subtitle: Some notes on a computational perspective on cognitive science.
---

{:toc}

# nativism, empiricism

- cogsci "inverse problem" - give world, how do we form representations
  - e.g. given pixels, how do we understand world?
- historical cognitive development: how do we form representations?
  - **nativism** - plato - representation w/out learning
  - **empiricism** - aristotle - learning w/out representation
  - **constructivism** - jean piaget - cognitive development
    - types
      - assimilation - start with schema, which tells you how to act
      - accomodation - adjust schema based on what you see
      - equilibration - everything set
    - periods
      | sensorimotor  | preoperational      | concrete operational  | formal operational     |
      | ------------- | ------------------- | --------------------- | ---------------------- |
      | 0 - 18 months | 18 months - 5 years | 5 years - adolescence | adolescence and beyond |
    - problems
      - dev. is domain-specific and variable
      - when measured better, children show earlier competence
      - doesn't specify learning methods
- contemporary theories
  - **nativism** - core knowledge or modules
    - plato, descartes
    - chomsky - language acuquisition device
    - spelke, tenenbaum - core knowledge of domains
    - constraint nativism vs. starting-state nativism
  - **empiricism** - connectionism, dynamic systems, associationism
    - aristotle, david hume
    - behaviorism - led to rl
    - connectionists - mclelland, karmiloff-smith
    - dynamic systems - thelen and smith
    - **emergentist appproach** - complex behavior is emergent from simple neural properties
    - john locke
    - deep learning
  - **constructivism** - "the theory theory" (children are scientists), probabilistic models
    - carey, wellman, gelman, gopnik
    - structural features: abstract, coherent, causal, hierarchical
    - theories change in response to evidence
    - changes may take place at multiple levels
    - playing
    - perhaps bayesian learning
  - information-processing - memory changes over time etc.
    - siegler
  - socio-cultural influences - children learn from society / culture
    - vyogotsky
- unanswered questions
  - search problem: how do children search through all possible hypotheses? sampling?
  - conceptual change problem: how is radical change in representations possible
- **counterfactual** - relating to or expressing what has not happened or is not the case
  - e.g. If kangaroos had no tails, they would topple over

# probabilistic causal models

- abstract representations that provide computational accounts of causal inference
  - at marr's highest level: computation
- how much do we need to build in?
  - dna can build in things that are hard to learn
  - start with nothing built in, ex. deep learning (connectionism)
  - start with best possible learning algorithm and ask what you need to build in (bayesian modeling)
  - bayes rule: $$\overbrace{p(\theta \vert  x)}^{\text{posterior}} = \frac{\overbrace{p(x\vert \theta)}^{\text{likelihood}} \overbrace{p(\theta)}^{\text{prior}}}{p(x)}$$ where x is data and $\theta$ are hypotheses
    - ask what priors explain people's inferences
    - humans make very causal priors - restricts hypothesis space of possible bayesian networks
    - hierarchical model - get prior from something (e.g. know all bags contain same color)
- what develops over time?
  - bayesian doesn't really tell us this - just has probabilities evolve over time
  - real life we come up with new hypotheses
- what representations do we use?

# deep rl

## dl basics

- what does an NN do?
  - universal function approximator
  - feature learning view - learn features that separate classes linearly
- [bhand...saxe, ng 2010](https://proceedings.neurips.cc/paper_files/paper/2011/hash/e19347e1c3ca0c0b97de5fb3b690855a-Abstract.html) compared statistics of primary cortical receptive fields and receptive field plasticity
  - vision + audio + somatosensory - statistics seem to line up with neuroscience
- a single algorithm?
  - ex. seeing with your tongue
  - ex. blind people echolocating
  - ex. ferret optic nerve rewired to auditory system and they learn to still see
- lots of inductive bias? or lack of bias?
  - gabor filters come from deep learning, ica, sparse coding, k-means - come from data, not necessarily algorithm
    - maybe this is only true for gabor filters though
  - bigger and simpler with residuals seems to work best
    - sequence tasks - lstm started being replaced with "attention is all you need" (attention is basically a dot product)
    - differentiable neural computer didn't seem to work all that well - tried to build in too much
    - some inductive bias has worked: cnn, lstm
- structure
  - structure is very dependent on optimization
- depth seems to help
  - compositionality
  - increase in representational power?
  - information theoretic arguments
- low-data regimes: few-shot recognition doesn't work (see Lake, Salakhutidinov, Tenenbaum 2013)
  - one type of meta-learning for few-shot learning
    - split data set into tiny train / test blocks and learn to do few-shot learning on these
    - got really good at this (especially for omniglot)
- deep learning: less structure + more data = better results
- alternative view - not necessarily dl
  - very unconstrained function class + right learning algorithm + right supervision = better results
  - optimization methods master (is sgd "conveniently" Bayesian?)
    - ex. everything that works works because it's Bayesian blog post
  - structure of natural dat amatters
    - source of supervision really matters
  - maybe function class doesn't really matter

## supervision

- unsupervised learning: learn $p_\theta (x)$
  - overview
    - pull out features/representation and use them for other tasks
    - recognize novel / unexpected events
    - hallucinate
    - can use all available data
    - often has principled, prob. interpretation
    - hard to use effectively
  - deep belief networks (~2006) - built on restricted Boltzmann machine (both a NN and a graphical model)
    - RBM = markov field with binary variables and connections between but not within layers
    - originally trained layerwise
    - properties
      - each neuron is (binary) random variable, so we can sample
      - hard to train
  - unsupervised gd (2012-2014) - sparse/denoising/bottleneck autoencoder
    - ex. if you have linear encoder/decoder with bottleneck, then you get pca
    - ex. denoising - blur image and ask it to learn to clean it up
      - learns something about the manifold that represents a class / the correct data
    - ex. variational autoencders (Kingma & Welling 2013)
      - network encoder x->z & decoder z->x
      - force z to be spherical Gaussian (ex. add noise to make it spherical Gaussian (with learned variance) - KL divergence regularizer between output and spherical Gaussian)
      - then, we can sample z from spherical Gaussian and decoder will give us nice looking x
    - ex. GANs (Goodfewllow 2014)
      - "implicit" density model
- supervised learning
  - representations are surprisingly good
    - labels provide good semantic knowledge
    - simple backprop training
  - semi-supervised = self-supervised learning
    - ex. given patches, tell where they are placed relative to each other (context prediction doersch et al. 2015)
    - prediction supervised learning (time series)
      - ex. predict video frames from past video frames
    - reakes some artistry to figure out objective
- reinforcement learning
  - non-differentiable reward function - doesn't access to output supervision
  - supervised learning is a subset of this
  - 3 types (often we do a combination of these together)
    - direct policy search
    - prediction of value functions
    - prediction of future states
  - can collect data by just exploring world
  - rl needs more generalization
    - want more exploration algorithms
    - train in more settings
    - ex. approximate bayesian experiment design

# objects

- *deep deep cnns* - not just a chain, combine different layers and maybe represent semantic concepts
  - *iterative deep aggregation* - generalizes skip connections - new layers for the skips
- *adaptive learning* - too much dataset bias - imagenet things only work on imagenet
  - need to know about discrepancy between distributions
  - GAN - learn adversarial that decides whether a point comes from one domain or another
  - goal: want both domains to be separable with domains for only one
- *explainable vision and language agents*
  - image captioning doesn't mean it actually understands scene
  - now, people focused on visual question answering
    - maybe even provide interpretable explanation
  - preprogram 5 types of modules (ex. find, relate, count, ...)
- concept vs. category learning
  - except for NN, need negative examples
  - learning nouns
    - strong sampling - pick key examples instead of lots of random examples
  - bayesian learning can figure out what classes to generalize to
    - need to assume some ontology
- it's all about the data
  - face detection - made things really work
  - learning spectrum: *extrapolation* (low samples) -> *interpolation* (high samples - where we are)
    - extrapolation = linear reg. -> interpolation = nearest neighbor / neural network
    - nearest neighbors started to work better with more data
  - explainability doesn't work if system is fundamentally complicated
  - natural world has too many examples for interpolation
  - brain doing nearest neighbors?
    - capacity of visual long term memory - standing 1973: 10k images, 83% recognition
    - clap if you see a repetition - you can really remember a lot
    - AB testing at end - check if you seaw things or not
    - novel objects: 92%, different examples of same thing: 88%, different states of same thing: 87%
    - we can't do this with just textured images though
    - big boosts come from data, not necessarily the algorithm
    - word2vec works about same as nearest neighbors embedding
- top-down vs bottom-up concept learning
  - problems with top-down / using labels
    - can't ask computer about semantic stuff - it only sees pixels
    - cool ex. can see a chair by just looking at silhouette of person sitting in it
    - humans don't categorize things in binary ways
      - solns: hierarchy, levels of categories...
    - used to have to do this (ex. books in a library), but computers don't have to do this (ex. amazon)
    - still problematic
      - intransitivity - car seat is chair, chair is furniture, ...
      - multiple category membership
      - "ontologies are overrated" blog post
    - this helps with interpretability, but might be worse overal
  - start from image (bottom) - use association not categorization
    - make graph - edges are associations
    - task: predict what's behind some blurred out box in an image


# language development

- lots of children have language difficulties
- language might be unique to humans
- language is somewhat instinctual
  - learned via association (John Locke) or by reading people's intentions (St. Augustine)
- behaviorist approach (Skinner) - conditioning, children learn because of adult reinforcement
  - ex. bird turning to get reward
  - "meaning" - an association between stimulus and response
  - progressilve taught more complex utterances
- noam chomsky's critique of skinner review (1959)
  - parents correct truth of utterance, not its grammar
  - stimulus doesn't completely determine response
  - children say things they've never heard (e.g. mommy eated the apple)
  - grammar isn't a chain of associated words
- chomsky: children have innate knowledge
  - introduces "cognitive revolution" - explain language you need
    - mental representations
    - rules that operate on those representations
  - figure out what is common between all languages
    - ex. verb agrees with subject
    - mostly part of speaker's unconscious knowledge of language
  - focus on ideal speaker's competence, not their performance
- ex. question formation
  - *poverty of the stimulus* - children get this info which doesn't seem to be in the stimulus
- chomsky theory of universal grammar
  - fixed invariant structural principles because human brain is wired to understand them
- *nativists* - children guided by universal grammar
- *constructivists* - innate **general learning mechanisms** operate on experience
- children use babbling
- children use pre-linguistic communication for sharing attention
- some children have crib speech - talking to themselves, seems systematic
- children have U-curves - use ate, then eated, then back to ate

# theory of mind

- Theory of mind = The ability to understand and take into account another individual’s mental state
- theory of mind - human understanding of agents' mental states and how they shape actions
  - understand intentional actions - like dots, understand evil, purpose
  - we can track this month by month
- Theory of mind
  - is rapidly acquired in the normal case
  - is acquired in an extended series of developmental accomplishments
  - encompasses several basic insights that are acquired world-wide on a roughly similar trajectory (but not timetable), (4) requires considerable learning and development based on an infant set of specialized abilities to attend to and represent persons, (5) is severely impaired in autism, (6) is severely delayed in deaf children of hearing parents, and (6) results from but also contributes to specialized neural substrates associated with reasoning about agency, experience, and mind
- Preschool Theory of Mind, Part 1: Universal Belief-Desire Understanding

# representations

based on berkeley's cs 188 course and russell & norvig's "artificial intelligence" (3rd edition)

## intro

- AI - field of study which studies the goal of creating intelligence
  - *intelligent agent* - system that perceives its environment and takes actions that maximize its chances of success
- expert task examples - medical diagnosis, equipment repair, computer configuration, financial planning

1. formal systems - use axioms and formal logic
2. *ontologies* - structuring knowledge in graph form
3. statistical methods

- *turing test* - is human mind deterministic { turing1950computing }
- *chinese room argument* - rebuts turing test { cite searle1980minds }
- *china brain* - what if different people hit buttons to fire individual neurons
- Polanyi's paradox: "we can know more than we can tell"

## symbol search

- computer science - empirical inquiry

### symbols and physical symbol systems
- intelligence requires the ability to store and manipulate symbols
- laws of qualitative structure
  - cell doctrine in biology
  - plate tectonics in geology
  - germ theory of disease
  - doctrine of atomism
- "physical"
  1. obey laws of physics
  2. not restricted to human systems
  - *designation* - then given the expression, the system can affect the object
  - *interpretation* - expression designates a process

### heuristic searching
- symbol systems solve problems with *heuristic search*
- *Heuristic Search Hypothesis* - solutions are represented as symbol structures. A physical symbol system exercises its intelligence in problem solving by search--that is, by generating and progressively modifying symbol structures until it produces a solution structure
    - from { cite newell1976computer }
- there are practical limitations on how fast computers can search
- To state a problem is to designate
    1. a test for a class of symbol structures (solutions of the problem)
    2. a generator of symbol structures (potential solutions).
- To solve a problem is to generate a structure, using (2), that satisfies the test of (1).
- searching is generally in a tree-form

## knowledge representation

- *physical symbol system hypothesis* - a physical symbol system has the necessary and sufficient means for general intelligent action
  - computers and minds are both *physical symbol systems*
  - *symbol* - meaningful pattern that can be manipulated
  - symbol system - creates, modifies, destroys symbols
- want to represent
  1. *meta-knowledge* - knowledge about what we know
  2. *objects* - facts
  3. *performance* - knowledge about how to do things
  4. *events* - actions
- two levels
  1. knowledge level - where facts are described
  2. symbol level - lower
- properties
  1. representational adequacy - ability to represent
  2. inferential adequacy
  3. inferential efficiency
  4. acquisitional efficiency - acquire new information
- two views of knowledge
  1. logic
     - a *logic* is a language with concrete rules
     - *syntax* - rules for constructing legal logic
     - *semantics* - how we interpret / read
       - assigns a meaning
     - multi-valued logic - not just booleans
     - higher-order logic - functions / predicates are also objects
     - multi-valued logics - more than 2 truth values
       - fuzzy logic - uses probabilities rather than booleans
     - match-resolve-act cycle
  2. associationist
    - knowledge based on observation
      - semantic networks - objects and relationships between them		- like is a, can, has
      - *graphical representation*
      - equivalent to logical statements
      - ex. nlp - conceptual dependency theory - sentences with same meaning have same graphs
      - *frame representations* - semantic networks where nodes have structure
        - ex. each frame has age, height, weight, ...
      - when agent faces *new situation* - slots can be filled in, may trigger actions / retrieval of other frames
      - inheritance of properties between frames
      - frames can contain relationships and procedures to carry out after various slots filled
- statistical
  - distributed - usually different from sparse code (sparser generally less robust)
    - opposite of sparse code = dense code
    - have to check multiple indexes
    - penti's work: distributed
    - usually want these to be robust
    - nlp is main place where unsupervised pretraining widely used
  - hierarchical
  - good representations - *linearly separable*
  - representation that *factors*
  - information bottleneck method: want simple representation that keeps class but throws away lots of extraneous info

## expert systems
- *expert system* - program that contains some of the subject-specific knowledge of one or more human experts.
- problems
  1. planning
  2. monitoring
  3. instruction
  4. control
- need lots of knowledge to be intelligent
- *rule-based architecture* - condition-action rules & database of facts
- acquire new facts
  - from human operator
  - interacting with environment directly
- forward chaining
  - until special HALT symbol in DB, keep following logical rule, add result to DB
- conflict resolution - which rule to apply when many choices available
- *pattern matching* - logic in the if statements
- *backward chaining* - check if something is true
  - check database
  - check if on the right side of any facts
- *CLIPS* - expert system shell
  - define rules and functions...
- *explanation subsystem* - provide explanation of reasoning that led to conclusion
- people
  1. *knowledge engineer* - computer scientist who designs / implements ai
  2. *domain expert* - has domain knowledge
1. user interface
2. knowledge engineering - art of designing and building expert systems
  - determine characteristics of problem
  - *automatic knowledge-acquisition* - set of techniques for gaining new knowledge
    - ex. parse Wikipedia
    - crowdsourcing
- creating an expert system can be very hard
  - only useful when expert isn't available, problem uses symbolic reasoning, problem is well-structured
- *MYCIN* - one of first successful expert systems { cite shortliffe2012computer }
  - Stanford in 1970s
  - used backward chaining but would ask patient questions - sometimes too many questions
- advantages
  - can explain reasoning
  - can free up human experts to deal with rare problems

## Godel, Escher, Bach
**Douglas Hofstadter, 1979**

- **meta**
  - *strange loop* = paradox - self-referential
  - zen enlightenment
    - goal: transcend **dualism** = division into concepts (perception, words do this)
    - words give you some truth but always fail to describe some parts of the truth
- **music**
  - **canons** - repeat w/ subtle changes (e.g. pitch shift)
  - **fugue** - repeat w/ more substantial changes
- **ai**
  - essential abilities - **can we do these things unsupervised?**
    - to recognize the relative importance of different elements of a situation
    - to find similarities between situations despite differences which may separate them;
    - to draw distinctions between situations despite similarities which may link them
    - to synthesize new concepts by taking old concepts and putting them together in new ways
  - intelligence consists of rules at different levels
    - "just plain" rules - like reflexes which respond to stereotyped situations
    - metarules - when situations are mixtures of steretoyped situations, requires rules for deciding which "just plain" rules to apply
    - rules for inventing new rules - when situations can't be classified
      - rules may have to change themselves
  - messages - comparison of DNA to a jukebox
    - where is info stored? records? buttons? smashed buttons?
    - what could constitute a Rosetta stone for DNA codes?
    - 3 parts
      - frame - tells you that this is a message
      - outer - tells you how to read a message (e.g. language, style)
      - inner - actual content
    - if decoding is universal, we might call the outer message (e.g. the trigger) the message
  - memory - same bits can be used for different things - part of each message specifies the instruction type
- **brain**
  - intelligence involves a calculus of descriptions = symbols
    - symbols represent both classes + instances (maybe both depending on amount of activation / context) - def need some context
    - can have links to other symbols (+priors on these)
    - *top-down logical structure*??
    - different ways to combine symbols get blurry
      - symbols can be learned to branch, merge
    - can harness temporal firing rates to encode more
    - can grow incrementally (greedily)
  - analogy of thoughts as trips on a (poorly fleshed out) map
- **interpretation**
  - ex. top is decimal expansion of the sum of the second ($\pi/4$)
    - 7, 8, 5, 3, 9, 8, 1, 6, ...
    - 1, -1/3, +1/5, -1/7, +1/9, -1/11...
- **math / logic**
  - **godel's thm** - limitation of any formal axiomatic system: cannot make a program to find a complete + consistent set of axioms
  - **church-turing thesis** - a function on the natural numbers can be calculated by an effective method if and only if it is computable by a Turing machine
    - no system can do computation which cannot be broken down into simple elements
  - **decision procedure** - decides whether something is a theorem - must terminate
  - we can think of theorems as strings in a formal system
  - **interpretation** - correspondence between symbols and words
    - ideally, these are meaningful isomorphisms between codes and reality
    - not all interpretations imply meaningful (or valid) corresponding codes
    - there might be multiple, equally valid interpretations
    - consistency depends on interpretation:
    - **consistency** - when every theorem, upon interpretation, comes out true (in some imaginable world)
    - **completeness** - when all statements which are true (in some imageinable world), and which can be expressed as well-formed strings of the system, are theorems
  - slightly different axioms lead to elliptical/hyperbolic geometry instead of Euclidean geometry
  - godel numbering - can replace all symbols w/ numbers and all typographic rules w/ arithmetic rules
  - 2 key idesas
    - strings can speak about other strings
    - self-scrutiny can be entire concentrated into a single string
  - every aspect of thinking can be viewed as a high-level description of a system which, on a low level, is governed by simple, even formal rules
- **causality**
  - what counterfactuals are the most realistic
    - different things are stable at different levels
- **biology**
  - dna -> rna -> proteins = sequence of amino acids
    - folds w/ valrious levels of structure (like music)
  - self-rep - what counts?
    - quine? instructions on jukebox? human reproduction?

# lesswrong

- [biases](https://www.lesswrong.com/s/5g5TkQTe9rmPS5vvM/p/ptxnyfLWqRZ98wnYi)
  - cognitive bias can be thought of like statistical bias
  - *base rate neglect*: grounding one’s judgments in how well sets of characteristics feel like they fit together, and neglecting how common each characteristic is in the population at large
    - e.g. if you meet a shy person, are they more likely to be a salesperson or a librarian?
  - *sunk cost fallacy*
  - *scope neglect* - the number of birds saved—the *scope* of the altruistic action—had little effect on willingness to pay ([post](https://www.lesswrong.com/s/5g5TkQTe9rmPS5vvM/p/2ftJ38y9SRBCBsCzy))
  - *availability heuristic* - judging the frequency or probability of an event by the ease with which examples of the event come to mind
    - [absurdity bias](https://www.lesswrong.com/lw/j4/absurdity_heuristic_absurdity_bias/); events that have never happened are not recalled, and hence deemed to have probability zero.
  - *conjunction fallacy* - humans assign a higher probability to a proposition of the form “A and B” than to one of the propositions “A” or “B” in isolation
    - the implausibility of one claim is compensated by the plausibility of the other; they “average out.”
  - *planning fallacy* - people think they can plan e.g. "best guess" scenarios are same as "best case" scenarios
- rationality
  - **epistemic rationality** - systematically improving the accuracy of your beliefs
  - **instrumental rationality** - systematically achieving your values
- System 1 and System 2—fast perceptual judgments versus slow deliberative judgments. System 2’s deliberative judgments aren’t always true, and System 1’s perceptual judgments aren’t always false; so it is very important to distinguish that dichotomy from “rationality.”
