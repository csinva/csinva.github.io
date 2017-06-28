---
layout: notes
section-type: notes
title: Research main
category: research
---

# to discuss
- prelim exam
- switching to cs dept?
- “the great events of the world take place in the brain” – Picture of Dorian Gray- Problems: Alzheimers, PTSD, autism, addiction, MS, depression, schizophrenia
- extracting memory with deep learning
	- learning how to find the right segments of memory
	- learning to decode another neural network?

# 1 - neural decoding
## fMRI decoding
- reconstructing visual experiences from brain activity evoked by movies (Gallant, 2011)
	- try doing this with music?
- typing
	- like fb or neuralink
- new data (e.g. BMI?)

## spike sorting
- can be based on electrodes or calcium-imaging data
- can get gt with intracellular recordings
- spike sorting with GANs
- simulated datasets also work well
- http://spikefinder.codeneuro.org

## neural encoding
- cochlear implant turns sound into neural signal
	
# 2 - brain mapping
## structural connectomics
- random forests / CNNs for neuronal image segmentation
- uses gradient boosting with MALIS

## functional connectivity
- computational fMRI (Cohen et al. 2017)
- using graphical models with weighted-l1 regularization

# 3 - computational learning models
## neural priors
- cox train cnn w/ fMRI

## comparison to cnn
- look at features found in layers

## biophysically plausible network learning
- PCA
- anti-hebbian learning (Foldiak)
- sparse coding (Olshausen & Field)
- ICA (Sejnowski)
- adaptive synaptogenesis with inhibition

# 4 - theoretical neural coding
## action potentials
- velocity vs. energy

## linearization
- linearization PLOS
- linearization JCNeuro
- interspike interval