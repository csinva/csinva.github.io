---
layout: notes_without_title
section-type: notes
title: dl for neuro
category: research
---

# Ideas for neuroscience using deep learning

Modern deep learning evokes many parallels with the human brain. Here, we explore how these two concepts are related and how deep learning can help understand neural systems using big data.

- https://medium.com/the-spike/a-neural-data-science-how-and-why-d7e3969086f2

# Brief history

The history of deep learning is intimately linked with neuroscience, with the modern idea of convolutional neural networks dates back to the necognitron<dt-cite key="fukushima1982neocognitron"></dt-cite>.

## pro big-data

Artificial neural networks can compute in several different ways. There is some evidence in the visual system that neurons in higher layers of visual areas can, to some extent, be predicted linearly by higher layers of deep networks<dt-cite key="yamins2014performance"></dt-cite>. However, this certainly isn't true in general.

- when comparing energy-efficiency, must normalize network performance by energy / number of computations / parameters

## anti big-data

- could neuroscientist  understand microprocessor
- no canonical microcircuit

# Data types

|              | EEG      | ECoG              | Local Field potential (LFP) -> microelectrode array | single-unit | calcium imaging | fMRI     |
| ------------ | -------- | ----------------- | --------------------------------------------------- | ----------- | --------------- | -------- |
| scale        | high     | high              | low                                                 | tiny        | low             | high     |
| spatial res  | very low | low               | mid-low                                             | x           | low             | mid-low  |
| temporal res | mid-high | high              | high                                                | super high  | high            | very low |
| invasiveness | non      | yes (under skull) | very                                                | very        | non             | non      |

- cellular
  - extracellular microeelectrodes
  - intracellular microelectrode
  - **neuropixels**
- optical
  - calcium imaging / fluorescence imaging
  - whole-brain light sheet imaging
  - voltage-sensitive dyes / voltage imaging
  - **adaptive optics**
  - fNRIS - like fMRI but cheaper, allows more immobility, slightly worse spatial res
  - **oct** - noninvasive - can look at retina (maybe find biomarkers of alzheimer's)
  - fiber photometry - optical fiber implanted delivers excitation light
- alteration
  - optogenetic stimulation
  - tms
    - genetically-targeted tms: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4846560/
  - local microstimulation with invasive electrodes
- high-level
  - EEG/ECoG
  - MEG
  - fMRI/PET
    - molecular fmri (bartelle)
  - MRS
  - event-related optical signal = near-infrared spectroscopy
- implantable
  - neural dust

# general projects

- could a neuroscientist understand a deep neural network? - use neural tracing to build up wiring diagram / function
- prediction-driven dimensionality reduction
- deep heuristic for model-building
- joint prediction of different input/output relationships
- joint prediction of neurons from other areas

# datasets

- MRNet: knee MRI diagnosis
- [datalad lots of stuff](http://datalad.org/datasets.html)
- springer 10k calcium imaging recording: https://figshare.com/articles/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622 

  - springer 2: 10k neurons with 2800 images

  - stringer et al. data

  - - 10000 neurons from visual cortex
- predicting running, facial cues
    - dimensionality reduction
		- enforcing bottleneck in the deep model
      - how else to do dim reduction?
- responses to 2800 images
- overview: http://www.scholarpedia.org/article/Encyclopedia_of_computational_neuroscience
- keeping up to date: https://sanjayankur31.github.io/planet-neuroscience/
- *lots of good data*: http://home.earthlink.net/~perlewitz/index.html
- connectome

  - fly brain: http://temca2data.org/
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



## interesting misc papers

- berardino 17 eigendistortions
  - **Fisher info matrix** under certain assumptions = $Jacob^TJacob$ (pixels x pixels) where *Jacob* is the Jacobian matrix for the function f action on the pixels x
  - most and least noticeable distortion directions corresponding to the eigenvectors of the Fisher info matrix
- gao_19_v1_repr
  - don't learn from images - v1 repr should come from motion like it does in the real world
  - repr
    - vector of local content
    - matrix of local displacement
  - why is this repr nice?
    - separate reps of static image content and change due to motion
    - disentangled rotations
  - learning
    - predict next image given current image + displacement field
    - predict next image vector given current frame vectors + displacement
- kietzmann_18_dnn_in_neuro_rvw
- friston_10_free_energy
  - ![friston_free_energy](data_ovw/friston_free_energy.png)



# deeptune-style

- ponce_19_evolving_stimuli: [https://www.cell.com/action/showPdf?pii=S0092-8674%2819%2930391-5](https://www.cell.com/action/showPdf?pii=S0092-8674(19)30391-5)
- bashivan_18_ann_synthesis
- [adept paper](https://papers.nips.cc/paper/6738-adaptive-stimulus-selection-for-optimizing-neural-population-responses.pdf)
  - use kernel regression from CNN embedding to calculate distances between preset images
  - select preset images
  - verified with macaque v4 recording
  - currently only study that optimizes firing rates of multiple neurons
	- pick next stimulus in closed-loop ("adaptive sampling" = "optimal experimental design")
- J. Benda, T. Gollisch, C. K. Machens, and A. V. Herz, “From response to stimulus: adaptive sampling in sensory physiology”
  - find the smallest number of stimuli needed to fit parameters of a model that predicts the recorded neuron’s activity from the
    stimulus

  - maximizing firing rates via genetic algorithms

  - maximizing firing rate via gradient ascent
- C. DiMattina and K. Zhang,“Adaptive stimulus optimization for sensory systems neuroscience"](https://www.frontiersin.org/articles/10.3389/fncir.2013.00101/full)

  - 2 general approaches: gradient-based approaches + genetic algorithms
  - can put constraints on stimulus space
  - stimulus adaptation
  - might want iso-response surfaces
  - maximally informative stimulus ensembles (Machens, 2002)
  - model-fitting: pick to maximize info-gain w/ model params
  - using fixed stimulus sets like white noise may be deeply problematic for efforts to identify non-linear hierarchical network models due to continuous parameter confounding (DiMattina and Zhang, 2010) 
  - use for model selection

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
</script>