---
layout: notes_without_title
section-type: notes
title: scattering transform
category: research
---


[TOC]



- some of the researchers involved
  - edouard oyallan, joan bruna, stephan mallat, Helmut Bölcskei, max welling

# goals

- benefits        
   - all filters are defined
   - more interpretable
   - more biophysically plausible
- scattering transform - computes a translation invariant repr. by cascading wavelet transforms and modulus pooling operators, which average the amplitude of iterated wavelet coefficients

# review-type
- mallat_16 "Understanding deep convolutional networks"
- vidal_17 "Mathematics of deep learning"
- bronstein_17 "Geometric deep learning: going beyond euclidean data"

# initial papers

- bruna_10 "classification with scattering operators"
- mallat_10 "recursive interferometric repr."
- mallat_12 "group invariant scattering"
  - introduce scat transform
- oyallan_13 "Generic deep networks with wavelet scattering"
- bruna_13 "Invariant Scattering Convolution Networks"
   - introduces the scattering transform implemented as a cnn
- anden_14 "Deep scattering spectrum"


## scat_conv
- oyallan_15 "Deep roto-translation scattering for object classification"
    - can capture rounded figures
    - can further impose robustness to rotation variability (although not full rotation invariance)
- cotter_17 "[Visualizing and improving scattering networks](https://arxiv.org/pdf/1709.01355.pdf)"
  - add deconvnet to visualize
- oyallan_18 "[Scattering Networks for Hybrid Representation Learning](https://hal.inria.fr/hal-01837587/document)"
    - using early layers scat is good enough
- oyallan_18 "i-RevNet: Deep Invertible Networks"
- oyallan_17 "[Scaling the scattering transform: Deep hybrid networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Oyallon_Scaling_the_Scattering_ICCV_2017_paper.pdf)"
    - use 1x1 convolutions to collapse accross channels
- jacobsen_17 "Hierarchical Attribute CNNs"
    - modularity
- cheng_16 "Deep Haar scattering networks"

## neuro style

- https://arxiv.org/pdf/1809.10504.pdf

## papers by other groups

- cohen_16 "[Group equivariant convolutional networks](http://www.jmlr.org/proceedings/papers/v48/cohenc16.pdf)"
  - introduce G-convolutions which share more wieghts than normal conv
- worrall_17 "[Interpretable transformations with encoder-decoder networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Worrall_Interpretable_Transformations_With_ICCV_2017_paper.pdf)"
  - look at interpretability
- bietti_17 "[Invariance and stability of deep convolutional representations](http://papers.nips.cc/paper/7201-invariance-and-stability-of-deep-convolutional-representations)"
  - theory paper
- cotter_18 "[Deep learning in the wavelet domain](https://arxiv.org/pdf/1811.06115.pdf)"

## wavelet style transfer

- [Photorealistic Style Transfer via Wavelet Transforms>](https://arxiv.org/pdf/1903.09760v1.pdf)

# helmut lab papers

- wiatoski_15 "[Deep Convolutional Neural Networks Based on Semi-Discrete Frames"](https://arxiv.org/abs/1504.05487)
  - allowing for different and, most importantly, general semidiscrete frames (such as, e.g., Gabor frames, wavelets, curvelets, shearlets, ridgelets) in distinct network layers
  - translation-invariant, and we develop deformation stability results
- wiatoski_18 "[A mathematical theory of deep convolutional neural networks for feature extraction](https://ieeexplore.ieee.org/document/8116648/)"
  - encompasses general convolutional transforms - general semi-discrete frames (including Weyl-Heisenberg filters, curvelets, shearlets, ridgelets, wavelets, and learned filters), general Lipschitz-continuous non-linearities (e.g., rectified linear units, shifted logistic sigmoids, hyperbolic tangents, and modulus functions), and general Lipschitz-continuous pooling operators emulating, e.g., sub-sampling and averaging
    - all of these elements can be different in different network layers. 
  - translation invariance result of vertical nature in the sense of the features becoming progressively more translation-invariant with increasing network depth
  - deformation sensitivity bounds that apply to signal classes such as, e.g., band-limited functions, cartoon functions, and Lipschitz functions.
- wiatowski_18 "[Energy Propagation in Deep Convolutional Neural Networks](https://arxiv.org/pdf/1704.03636.pdf)"

# nano papers

- yu_06 "A Nanoengineering Approach to Regulate the Lateral Heterogeneity of Self-Assembled Monolayers"
  - regulate heterogeneity of self-assembled monlayers
    - used nanografting + self-assembly chemistry
- bu_10 nanografting - makes more homogenous morphology
- fleming_09 "dendrimers"
  - scanning tunneling microscopy - provides highest spatial res
  - combat this for insulators
- lin_12_moire
  - prob moire effect with near-field scanning optical microscopy
- chen_12_crystallization

#  wavelets intro

- notes based on “An intro to wavelets” - Amara 
- has frequency and duration
- convolve w/ signal and see if they match
- set of complementary wavelets - decomposes data without gaps/overlap so that decomposition process is reversible
- better than Fourier for spikes / discontinuities
- translation covariant (not invariant)

## Fourier transform

- translate function in time domain to frequency domain
- *discrete fourier transform* - estimates Fourier transform from a finite number of its sampled points
- *windowed fourier transform* - chop signal into sections and analyze each section separately
- *fast fourier transform* - factors Fourier matrix into product of few sparse matrices

## wavelet comparison to Fourier

- both linear operations
- the inverse transform matrix is transpose of the original
- both localized in frequency
- wavelets are also localized in space
- makes many functions sparse in wavelet domain
- Fourier just uses sin/cos
- wavelet has infinite set of possible basis functions

## wavelet analysis

- must adopt a wavelet prototype function $\phi(x)$, called an *analyzing wavelet*=*mother wavelet*
- orthogonal wavelet basis: $\phi_{(s,l)} (x) = 2^{-s/2} \phi (2^{-s} x-l)$
- *scaling function* $W(x) = \sum_{k=-1}^{N-2} (-1)^k c_{k+1} \phi (2x+k)$ where $\sum_{k=0,N-1} c_k=2, \: \sum_{k=0}^{N-1} c_k c_{k+2l} = 2 \delta_{l,0}$
- one pattern of coefficients is smoothing and another brings out detail = called *quadrature mirror filter pair*
- there is also a fast discrete wavelet transform (Mallat)
- wavelet packet transform - basis of *wavelet packets* = linear combinations of wavelets
- *basis of adapted waveform* - best basis function for a given signal representation
- *Marr wavelet* - developed for vision
- differential operator and capable of being tuned to act at any desired scale

## l2 functions

- $L^2$ function is a function $f: X \to \mathbb{R}$ that is square integrable: $|f|^2 = \int_X |f|^2 d\mu$ with respect to the measure $\mu$
  - $|f|$ is its $L^2$-norm
- **measure ** = nonnegative real function from a delta-ring F such that $m(\empty) = 0$ and $m(A) = \sum_n m(A_n)$ 
- **Hilbert space** H: a vectors space with an innor product $<f, g>$ such that the following norm turns H into a complete metric space: $|f| = \sqrt{<f, f>}$
- **diffeomorphism** is an [isomorphism](https://en.wikipedia.org/wiki/Isomorphism) of [smooth manifolds](https://en.wikipedia.org/wiki/Smooth_manifold). It is an [invertible function](https://en.wikipedia.org/wiki/Invertible_function) that [maps](https://en.wikipedia.org/wiki/Map_(mathematics)) one [differentiable manifold](https://en.wikipedia.org/wiki/Differentiable_manifold) to another such that both the function and its inverse are [smooth](https://en.wikipedia.org/wiki/Smooth_function).

# wavelet families

- [overview](https://www.eecis.udel.edu/~amer/CISC651/IEEEwavelet.pdf)
- [matlab wavelets](https://www.mathworks.com/help/wavelet/ug/wavelet-families-additional-discussion.html)
- [python wavelets](http://wavelets.pybytes.com/)
- morlet = gabor
- generalizations
  - [generalized daubechet wavelet families](http://bigwww.epfl.ch/publications/vonesch0702.pdf)
  - [generalized coiflets](https://pdfs.semanticscholar.org/46e3/4016b8c4b187118e83392242c2165a6db3db.pdf)
  - [Wavelet families of increasing order in arbitrary dimensions](https://ieeexplore.ieee.org/abstract/document/826784)
  - [Parametrizing smooth compactly supported wavelets](https://www.ams.org/journals/tran/1993-338-02/S0002-9947-1993-1107031-8/)
    - just for daubuchet

## reversible/invertible models

- [williams  inverting w/ autoregressive models](https://arxiv.org/abs/1806.00400)

- [arora reversibility](https://arxiv.org/pdf/1511.05653.pdf) 
- iResNet