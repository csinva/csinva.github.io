[TOC]

# success + challenges in modern ai

**amnon shashua, Hebrew University**

- 3 errs
  - training (optimization)
  - estimation (generalization)
  - approximation (expressiveness)
- dl we simultaneously reduce all
- autonomous driving
  - perception - want 1 err every $10^7​$ hours; use redundant sensors (e.g. lidar, cameras)
  - driving policy - hard to measure sucess
- nlp is future (NLU = natural language understanding)
  - SQuAD - BERT is achieving human-performance
  - all about "inductive bias" - how are assumptions about target function in architecture
    - cnns connectivity/pooling work great for pattern recognition
    - unclear what this is in NLP
  - reading comprehension is a good proxy - read a book and answer questions

# deep learning in computer vision

**jitendra malik, uc berkeley**

- evolutionary progression: vision + locomotion, manipulation, language (ai seems to solve in the same order as evolution)
- successes: cv, speech understanding, machine translation, game playing
- paradigms ~1960
  - classic ai (mccarthy, minsky, newell...)
  - pattern recognition (rosenblatt...)
  - estimation and control (bellman, kalman...)
- future
  - 3d vision
  - predicting the future
  - few-shot learning, little supervision
  - unifying learning with geometric reasoning
  - perception + control

# state of dl for nlp

**christopher manning, stanford**

- language is distinctive human characteristic, explanations, social intelligence, constructed to convey meaning
- language is discrete, encoded as continuous communication signal
- early breakthroughs: george dahlr 2010/2012 speech recognition
- noam chomsky: probability won't be helpful for language
- words: no inherent similarity
  - distributional semantics - word's meaning is given by distr. of nearby wrods
  - ex. word2vec, GloVe
- language models: predict the probability of next words
  - loss: next word pred prob should match what is seen
  - adding skip connection is effective: lstm, etc.
  - this allowed for machine translation (read in one language into hidden state then generate next language)
  - attention provided next big boost
- question answering - DrQA
- contextual word representations - words mean things in different contexts
  - train large unsupervised language models - might be good general representations
  - ex. ELMO, GPT, BERT, GPT-2
- transformer architectures - non-recurrent (faster), uses only attention

# state of deep rl

**orial vinyals, deepmind**

- atari is like mnist of cv
- alphago: prune search tree to just reasonable moves
- when does it work (convert into suprvised learning)
  - environment is full of rewards and random agent gets them sometimes (e.g. atari)
  - available human demonstrations at scale 
  - algorithmic ways to improve the policty (search for alphago)
  - you can simulate many trajectories fast
- real-world problems
  - ok to use simulation
  - things should transfer (imagenet -> mnist)
  - alphastar/alphago should learn chess quickly

# panel

- high-dimensional spaces are very non-intuitive and need mathematical understanding
- fake news dataset - didn't need evidence, just found negation

# building chemistry-aware neural models

**Regina Barzilay, MIT**

- drugs: search problem (people use some human intuition to guess)
- tasks
  - property prediction
  - reaction prediction
  - optimization for drug discovery
- open questions
  - molecular representation beyond graphs - want continuous latent space
  - modeling underlying physics

# dl in particle physics

**kyle cranmer, nyu**

- atlas experiment
- complex simulation-based - evaluation of likelihood is intractable
- likelihood-free methods = approximate bayesian computation
  - can only run model forward - how to get likelihood?
- use simulator
  - abc
  - probabilistic programming - distr over program examples
    - can force forward to match the thing you want to condition on by messing with it
  - adversarial variational optimization
- learn simulator
  - gans/vae
  - likelihood-ratio from classifiers (CARL)

# dl in genomics

**olga troyanskaya, princeton**

- searching genome for different predisposition to diseases
- predict the effect of any mutation - hard to do - no matched pairs

# deep learning and biological vision

**eero simoncelli, nyu**

- kriegeskorte - mri stimulus similarity (DNNs good for predicting neural activity)
- missing
  - largely unsupervised, non-classification
  - local learning/adaptation/gain-control/homeostasis
  - recurrence/state/context (e.g. memory, reward, attention)
- biological modelling: how simple can elements be?
- ex. quantifying perceptual distances (berardino et al. nips)
  - local gain control is important
  - standard choice: mse - fails because doesn't match nonlinear human visual system's distortion space
  - model: static gain control model of LGN (retina, luminance gain control, contrast gain control) ~ simple
  - all fit to TID-2008 human subjective perceptual loss data - about the same
  - for simple model, distortions which model says will be most visible/least visible are to humans
- ex. perceptual straightening of videos (henaff 19)
  - temporal straightening hypothesis - frames of movie might be a straight line in population of neurons
  - simple models perform appropriate straightening (lgn, v1)
  - dnn does not
- gain control vs batch normalization?
- wishlist???



# bruno

- bruno says read this: **o regan & noe 2001 - philosophy paper on motors/actuators**
- experiments to test this?

# torralba

- individual units semantic segmentation matches extremely well with certain classes
- find units in gan that correlate to classes (measured with segmentation)
  - can knock these out to manipulate gan

# solving 3 theoretical puzzles in deep nets

**tomas poggio, mit**

- approximation theory: when and why are deep nets better than shallow?
  - deep nets avoid the curse for compositional functions
- optimization: what is the landscape of the empirical risk
  - can nstudy polynomial nets (replaces relu)
  - in high dims, always find degenerate minima = global
- learning theory: how can deep learning generalize?
  - if you normalize, training loss predicts testing loss
  - try to explicitly optimize rademacher bound from bartlett
  - gd/sgd withe weight or batch normalization implement generalization bounds

# inductive bias + optimization in dl

**nati srebro, TTIC**

- 3 questions
  - capacity / generalization ability / sample complexity: ~number of weights
  - expressive power / approximation: any continuous function
    - lots of interesting things naturall w/ small networks
    - any time T computable function (test time takes T) with network of size $O(T)$
  - computation / optimization: np-hard to find weights even with 2 hidden units
    - no poly-time algorithm always works
    - **this is magic: what makes local search work**
- neyshabur et al. 15
  - as hidden units increases, train err=0, test err keeps decreasing
  - size of weights is more important that size of network (bartlett 97)
- what is relevant complexity measure (e.g. norm) that gives us test err?
- need to understand optimization alg. not just as reaching some (global) optimum, but as reaching a specific optimum
- numunits is like num bits in a float
- effect of parametrization: changes inductive bias, not expressive power
  - ex. gunasekar et al 2017: unconstrained matrix completion
    - grad descent on U, V yields min nuclear norm solution
  - ex. [soudry et al 2017](http://www.jmlr.org/papers/volume19/18-188/18-188.pdf)
    - sgd on logistic reg. gives hard margin svm
    - deep linear net gives the same thing - doesn't actually changed anything
  - ex. [gunaskar, 2018](http://papers.nips.cc/paper/8156-implicit-bias-of-gradient-descent-on-linear-convolutional-networks)
    - linear convnets give smth better - minimum l1 norm in discrete fourier transform 
  - ex. savarese 2019
    - infinite width relu net 1-d input
    - weight decay minimization minimizes derivative of TV
- **eero asks if init is important - nati says in high-dim no**

# accurate prediction from interpolation

**peter bartlett, uc berkeley**

- empirical process theory for classification
  - vc theory: depends on number of parameters
- margins analysis: relating classification to regression
  - neyshabur et al. 2015 generalizes rademacher averages
  - bartlett et al. 2017 can be made to work for non-sigmoid nets
- interpolation: tradeoff between fit and complexity
- interpolation in linear regression
  - interpolate means training error is 0
  - in high-dim, minimum norm interpolant can hide the noise in many unimportant directions
  - relies on overparameterization

# why neuro needs the science of deep learning

**konrad kording** (from could neuroscientist understand microprocessor paper)

- sexual evolution dramatically useful for evolution sample complexity
- dl algorithms are like evolution...random stuff now
- want to invert: go from inductive biases -> algo, currently goes other way
- goals of comp neuro
  - model must be understandable my humans
    - some info is uncompressible - can't be understood
    - we can't find dl recpies, just principles
  - model must work

# instabilities in deep learning

**anders hansen, cambridge**

- neural nets are unstable within epsilon ball around training data points
- however, there exists a network which is stable
- other functions can still be stable

# challenge + scope for empirical modelling by ml

**ronald coifman, yale**

- eigenvectors of nxn kernel matrices can be compared

# theory-based measures of object representations in anns/dnns

**haim sompolinsky, hebrew university of jerusalem**

- current methods: single-neurons, similarity matrices
- untangling perceptual manifolds (dicarlo & cox 07)
  - hypothesis: thse manifolds become linearly separable in high-level sensory representations
- manifold = compact bounded set of N dim vectors lying in a D+1 << N dim linear subspace
- tangled manifolds
  - overlapping
  - pairwise nonlinearly separable
  - manifold ensemble nonlinearly classifiable (can separate but not linearly)
- imagenet seems to look like ensemble of nonlinearly classifiable manifolds
- capacity = maximum number of separable manifolds, P, scales linearly with embedding dimensionality, N
  - **capacity = P/N**
- most important geometric parameters are manifold radius and dimension
- ex. geometric properties of face repr. in macaque face patchy system (cohen, taso 10)
- ex. object manifolds in macaque monkey (dapello, chung from dicarlo lab)
