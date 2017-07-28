---
layout: notes
section-type: notes
title: Research ref
category: research
---

# datasets
- senseLab: https://senselab.med.yale.edu/
	- modelDB - has NEURON code
- model databases: http://www.cnsorg.org/model-database 
- comp neuro databases: http://home.earthlink.net/~perlewitz/database.html
- crns data: http://crcns.org/
	- hippocampus spike train data: http://crcns.org/data-sets/hc
	- visual cortex data (gallant)
- allen brain atlas
- wikipedia page: https://en.wikipedia.org/wiki/List_of_neuroscience_databases
- *human fMRI datasets*: https://docs.google.com/document/d/1bRqfcJOV7U4f-aa3h8yPBjYQoLXYLLgeY6_af_N2CTM/edit
	- Kay et al 2008 has data on responses to images
- calcium imaging data: http://spikefinder.codeneuro.org/
- spikes: http://www2.le.ac.uk/departments/engineering/research/bioengineering/neuroengineering-lab/software

# data types
|  | fMRI | EEG | ECoG | Local field potential (together forms microelectrode array) | single-unit | calcium imaging |
|--------------|----------|----------|-------------------|-------------------------------------------------------------|-------------| ---|
| scale | high | high | high | low | tiny |
| spatial res | mid-low | very low | low | mid-low | x |
| temporal res | very low | mid-high | high | high | super high |
| invasiveness | non | non | yes (under skull) | very | very |
- neural dust

# ongoing projects
- gov-sponsored
	- human brain project
	- blue brain project	- large-scale brain simulation
	- european brain project
- companies
	- Neuralink
	- Kernel
	- Facebook neural typing interface
	- google brain
	- IBM: project joshua blue

#conferences: 
- Annual Computational Neuroscience Meeting
- Statistical Analysis of Neuronal Data
- 2017
    - SFN (11/11-11/15) - DC
    - NIPS (12/4-12/9) - Long Beach
- 2018
    - ICCV (March)
    - VSS (5/18-5/23) - Florida (Always)
    - ICML (7/10-7/15) - Stockholm
    - CVPR (6/18-6/23) - Salt Lake City
    - SFN (11/3-11/7) - San Diego
    - NIPS (12/3-12/8) - Montreal
- 2019
    - ICCV (March) - Korea?
    - ICML (7/10-7/14) - Long Beach
    - CVPR (Unknown)
    - SFN (10/19-10/23) - Chicago
    - NIPS (Unknown)

# areas
- Basic approaches:
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