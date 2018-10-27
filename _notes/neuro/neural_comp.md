---
layout: notes
section-type: notes
title: comp neuro 2
category: neuro
---

* TOC
{:toc}
---

- website: redwood website
- convs are organized spatially
- could do transform so that have spetial convs that are organized spatially, orientation based, frequency based
- projects
  - 5 page report
  - pres last day (oral pres - 10 min or poster) will go all day
  - should be new, different

# introduction

## overview

- does biology have a cutoff level (likecutoffs in computers below which fluctuations don't matter)
- core principles underlying these two questions
  - how do brains work?
  - how do you build an intelligent machine?
- lacking: insight from neuro that can help build machine
- scales: cortex, column, neuron, synapses
- physics: theory and practice are much closer
- are there principles?
  - "god is a hacker" - francis crick
  - theorists are lazy - ramon y cajal
  - things seemed like mush but became more clear - horace barlow
  - principles of neural design book
- felleman & van essen 1991
  - ascending layers (e.g. v1-> v2): goes from superficial to deep layers
  - descending layers (e.g. v2 -> v1): deep layers to superficial
- **solari & stoner 2011 "cognitive consilience"** - layers thicknesses change in different parts of the brain
  - motor cortex has much smaller input (layer 4), since it is mostly output

## historical ai

- people: turing, von neumman, marvin minsky, mccarthy...
- ai: birth at 1956 conference
  - vision: marvin minsky thought it would be a summer project
- lighthill debate 1973 - was ai worth funding?
- intelligence tends to be developed by young children...
- cortex grew very rapidly

## historical cybernetics/nns

- people: norbert weiner, mcculloch & pitts, rosenblatt
- neuro
  - hubel & weisel (1962, 1965) simple, complex, hypercomplex cells
  - neocognitron fukushima 1980
  - david marr: theory, representation, implementation

# neuron models

## circuit-modelling basics

- membrane has capacitance $C_m$
- force for diffusion, force for drift
- can write down diffeq for this, which yields an equilibrium
- $\tau = RC$
  - bigger $\tau$ is slower
  - to increase capacitance
    - could have larger diameter
    - $C_m \propto D$
  - axial resistance $R_A \propto 1/D^2$ (not same as membrane lerk), thus bigger axons actually charge faster

## action potentials

- channel/receptor types
  - ionotropic: $G_{ion}$ = f(molecules outside)
    - something binds and opens channel
  - metabotropic: $G_{ion}$ = f(molecules inside)
    - doesn't directly open a channel: indirect
  - others
    - photoreceptor
    - hair cell
  - voltage-gated (active - provide gain; might not require active ATP, other channels are all passive)

## physics of computation

- based on carver mead: drift and diffusion are at the heart of everything
- different things realted by the **Boltzmann distr.** (ex. distr of air molecules vs elevation. Subject to gravity and diffusion upwards since they're colliding)
  - nernst potential
  - current-voltage relation of voltage-gated channels
  - current-voltage relation of MOS transistor
- these things are all like transistor: energy barrier that must be overcome
- neuromorphic examples
  - differential pair sigmoid yields sigmoid-like function
    - can compute tanh function really simply to simulate
  - silicon retina
    - lateral inhibition exists (gap junctions in horizontal cells)
    - mead & mahowald 1989 - analog VLSI retina (center-surround receptive field is very low energy)
- computation requires energy (otherwise signals would dissipate)
  - von neumann architecture: CPU - bus (data / address) - Memory
    - moore's law ending (in terms of cost, clock speed, etc.)
      - ex. errors increase as device size decreases (and can't tolerate any errors)
  - neuromorphic computing
    - brain ~ 20 Watts
    - exploit intrinsic transistor physics (need extremely small amounts of current)
    - exploit electronics laws kirchoff's law, ohm's law
    - new materials (ex. memristor - 3d crossbar array)
    - can't just do biological mimicry - need to understand the principles

# supervised learning

- see machine learning course
- net talk was major breakthrough (words -> audio) Sejnowski & Rosenberg 1987
- people looked for world-centric receptive fields (so neurons responded to things not relative to retina but relative to body) but didn't find them
  - however, they did find gain fields: (Zipser & Anderson, 1987)
    - gain changes based on what retina is pointing at
  - trained nn to go from pixels to head-centered coordinate frame
    - yielded gain fields
  - pouget et al. were able to find that this helped having 2 pop vectors: one for retina, one for eye, then add to account for it
- support vector networks (vapnik et al.) - svms early inspired from nns
- dendritic nonlinearities (hausser & mel 03)
- example to think about neurons due this: $u = w_1 x_1 + w_2x_2 + w_{12}x_1x_2$
  - $y=\sigma(u)$
  - somestimes called sigma-pi unit since it's a sum of products
  - exponential number of params...**could be fixed w/ kernel trick?**
    - could also incorporate geometry constraint...

# unsupervised learning

- born w/ extremely strong priors on weights in different areas
- barlow 1961, attneave 1954: efficient coding hypothesis = redundancy reduction hypothesis
  - representation: compression / usefulness
  - easier to store prior probabilities (because inputs are independent)
  - relich 93: redundancy reduction for unsupervised learning (text ex. learns words from text w/out spaces)

## hebbian learning and pca

- pca can also be thought of as a tool for decorrelation (in pc dimension, tends to be less correlated)
- hebbian learning = fire together, wire together: $\Delta w_{ab} \propto <a, b>$ note: $<a, b>$ is correlation of a and b (average over time)
- linear hebbian learning (perceptron with linear output)
- $\dot{w}_i \propto <y, x_i> \propto \sum_j w_j <x_j, x_i>$ since weights change relatively slowly
  - synapse couldn't do this, would grow too large
- oja's rule (hebbian learning w/ weight decay so ws don't get too big)
  - points to correct direction
- sanger's rule: for multiple neurons, fit residuals of other neurons
- competitive learning rule: winner take all
  - population nonlinearity is a max
  - gets stuck in local minima (basically k-means)
- pca only really good when data is gaussian
  - interesting problems are non-gaussian, non-linear, non-convex
- pca: yields checkerboards that get increasingly complex (because images are smooth, can describe with smaller checkerboards)
  - this is what jpeg does
  - very similar to discrete cosine transform (DCT)
  - very hard for neurons to get receptive fields that look like this
- retina: does whitening (yields center-surround receptive fields)
  - easier to build
  - gets more even outputs
  - only has ~1.5 million fibers

# sparse, distributed coding

- barlow 1972: want to represent stimulus with minimum active neurons
  - neurons farther in cortex are more silent
  - v1 is highly overcomplete (dimensionality expansion)
- codes: dense -> sparse, distributed $n \choose k$ -> local (grandmother cells)
  - energy argument - bruno doesn't think it's a big deal (could just not have a brain)
- PCA: autoencoder when you enforce weights to be orthonormal
  - retina must output encoded inputs as spikes, lower dimension -> uses whitening
- cortex
  - sparse coding different kind of autencoder bottleneck (imposes sparsity)
- using bottlenecks in autoencoders forces you to find structure in data
- v1 simple-cell receptive fields are localized, oriented, and bandpass
- higher-order image statistics
  - phase alignment
  - orientation (requires at least 3 points stats (like orientation)
  - motion
- how to learn sparse repr?
  - foldiak 1990 forming sparse reprs by local anti-hebbian learning
  - driven by inputs and gets lateral inhibition and sum threshold
  - neurons drift towards some firing rate naturally (adjust threshold naturally)
- use higher-order statistics
  - projection pursuit (field 1994) - maximize non-gaussianity of projections
    - CLT says random projections should look gaussian
    - gabor-filter response histogram over natural images look non-Gaussian (sparse) - peaked at 0
  - doesn't work for graded signals
- sparse coding for graded signals: olshausen & field, 1996
  - $\underset{Image}{I(x, y)} = \sum_i a_i \phi_i (x, y) + \epsilon (x,y)$
  - loss function $\frac{1}{2} |I - \phi a|^2 + \lambda \sum_i C(a_i)$
  - can think about difference between $L_1$ and $L_2$ as having preferred directions (for the same length of vector) - prefer directions which some zeros
  - in terms of optimization, smooth near zero
  - there is a network implementation
  - $a_i$are calculated by solvin optimization for each image, $\phi$ is learned more slowly
  - **can you get $a_i$ closed form soln?** 
- wavelets invented in 1980s/1990s for sparsity + compression
- these tuning curves match those of real v1 neurons
- applications
  - for time, have spatiotemporal basis where local wavelet moves
  - sparse coding of natural sounds
    - audition like a movie with two pixels (each ear sounds independent)
    - converges to gamma tone functions, which is what auditory fibers look like
  - sparse coding to neural recordings - finds spikes in neurons
    - learns that different layers activate together, different frequencies come out
    - found place cell bases for LFP in hippocampus
  - nonnegative matrix factorization - like sparse coding but enforces nonnegative 
  - can explicitly enforce nonnegativity
- LCA algorithm lets us implement sparse coding in biologically plausible local manner
- explaining away - neural responses at the population should be decodable (shouldn't be ambiguous)
- good project: understanding properties of sparse coding bases
- SNR = $VAR(I) / VAR(|I- \phi A|)$
- can run on data after whitening
  - graph is of power vs frequency (images go down as $1/f$), need to weighten with f
  - don't whiten highest frequencies (because really just noise)
    - need to do this softly - roughly what the retina does
  - as a result higher spatial frequency activations have less variance
- whitening effect on sparse coding
  - if you don't whiten, have some directions that have much more variance
- projects
  - applying to different types of data (ex. auditory)
- adding more bases as time goes on
- combining convolution w/ sparse coding?
- people didn't see sparsity for a while because they were using very specific stimuli and specific neurons
  - now people with less biased sampling are finding more sparsity
  - in cortex anasthesia tends to lower firing rates, but opposite in hippocampus

# self-organizing maps

- homunculus - 3d map corresponds to map in cortex (sensory + motor)
- visual cortex
  - visual cortex mostly devoted to center
  - different neurons in same regions sensitive to different orientations (changing smoothly)
  - orientation constant along column
  - orientation maps not found in mice (but in cats, monkeys)
  - direction selective cells as well
- maps are plastic - cortex devoted to particular tasks expands (not passive, needs to be active)
  - kids therapy with tone-tracking video games at higher and higher frequencies

# recurrent networks

- hopfield nets can store / retrieve memories
- fully connected (no input/output) - activations are what matter
  - can memorize patterns - starting with noisy patterns can converge to these patterns
- marr-pogio stereo algorithm
- hopfield three-way connections
  - $E = - \sum_{i, j, k} T_{i, j, k} V_i V_j V_k$ (self connections set to 0)
    - update to $V_i$ is now bilinear
- dynamic routing
  - hinton 1981 - reference frames requires structured representations
    - mapping units vote for different orientations, sizes, positions based on basic units
    - mapping units **gate the activity** from other types of units - weight is dependent on if mapping is activated
    - top-down activations give info back to mapping units
    - this is a hopfield net with three-way connections (between input units, output units, mapping units)
    - reference frame is a key part of how we see - need to vote for transformations
  - olshausen, anderson, & van essen 1993 - dynamic routing circuits
    - ran simulations of such things (hinton said it was hard to get simulations to work)
    - we learn things in object-based reference frames
    - inputs -> outputs has weight matrix gated by control
  - zeiler & fergus 2013 - visualizing things at intermediate layers - deconv (by dynamic routing)
    - save indexes of max pooling (these would be the control neurons)
    - when you do deconv, assign max value to these indexes
  - arathom 02 - map-seeking circuits
  - tenenbaum & freeman 2000 - bilinear models
    - trying to separate content + style
  - hinton et al 2011 - transforming autoencoders - trained neural net to learn to shift imge
  - sabour et al 2017 - dynamic routing between capsules
    - units output a vector (represents info about reference frame)
    - matrix transforms reference frames between units
    - recurrent control units settle on some transformation to identify reference frame

# probabilistic models + inference



# boltzmann machines



# ica



# dynamical models



# neural coding



# high-dimensional computing

