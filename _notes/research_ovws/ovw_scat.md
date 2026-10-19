---
layout: notes
title: scattering transform
category: research
subtitle: Some papers involving the scattering transform and similar developments bringing structure to replace learned filters.
---

{:toc}

- some of the researchers involved
  - edouard oyallan, joan bruna, stephan mallat, Helmut Bölcskei, max welling

# goals

- benefits
   - all filters are defined
   - more interpretable
   - more biophysically plausible
- scattering transform - computes a translation invariant repr. by cascading wavelet transforms and modulus pooling operators, which average the amplitude of iterated wavelet coefficients

# review-type
- Understanding deep convolutional networks ([mallat 2016](https://arxiv.org/abs/1601.04920))
- Mathematics of deep learning ([vidal et al. 2017](https://arxiv.org/abs/1712.04741))
- Geometric deep learning: going beyond euclidean data ([bronstein et al. 2017](https://arxiv.org/abs/1611.08097))

# initial papers

- classification with scattering operators ([bruna & mallat 2010](https://arxiv.org/abs/1011.3023))
- recursive interferometric repr. ([mallat 2010](https://www.di.ens.fr/data/publications/papers/Eusipco2010InterConfPap.pdf))
- group invariant scattering ([mallat 2012](https://arxiv.org/abs/1101.2286))
  - introduces scat transform
- Generic deep networks with wavelet scattering ([oyallan et al. 2013](https://arxiv.org/abs/1312.5940))
- Invariant Scattering Convolution Networks ([bruna & mallat 2012](https://arxiv.org/abs/1203.1513))
   - introduces the scattering transform implemented as a cnn
- Deep scattering spectrum ([anden & mallat 2013](https://arxiv.org/abs/1304.6763))

## scat_conv

- Deep roto-translation scattering for object classification ([oyallan & mallat 2014](https://arxiv.org/abs/1412.8659))
    - use 1x1 conv on top of scattering coefs (only 1 layer)
    - can capture rounded figures
    - can further impose robustness to rotation variability (although not full rotation invariance)
    - Deep learning in the wavelet domain ([cotter & kingbury, 2017](https://arxiv.org/pdf/1811.06115.pdf))- each conv layer is replaced by scattering transform + 1x1 conv
- Visualizing and improving scattering networks ([cotter et al. 2017](https://arxiv.org/pdf/1709.01355.pdf))
  - add deconvnet to visualize
- Scattering Networks for Hybrid Representation Learning ([oyallon et al. 2018](https://hal.inria.fr/hal-01837587/document))
    - using early layers scat is good enough
- i-RevNet: Deep Invertible Networks ([jacobsen et al. 2018](https://arxiv.org/abs/1802.07088))
- Scaling the scattering transform: Deep hybrid networks ([oyallon et al. 2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Oyallon_Scaling_the_Scattering_ICCV_2017_paper.pdf))
    - use 1x1 convolutions to collapse accross channels
- jacobsen_17 "Hierarchical Attribute CNNs"
    - modularity
- cheng_16 "Deep Haar scattering networks"
- Deep Network Classification by Scattering and Homotopy Dictionary Learning ([zarka et al. 2019](https://arxiv.org/abs/1910.03561)) - scat followed by sparse coding then linear

## neuro style

- https://arxiv.org/pdf/1809.10504.pdf

## papers by other groups

- cohen_16 "[Group equivariant convolutional networks](http://www.jmlr.org/proceedings/papers/v48/cohenc16.pdf)"
  - introduce G-convolutions which share more wieghts than normal conv
- worrall_17 "[Interpretable transformations with encoder-decoder networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Worrall_Interpretable_Transformations_With_ICCV_2017_paper.pdf)"
  - look at interpretability
- bietti_17 "[Invariance and stability of deep convolutional representations](http://papers.nips.cc/paper/7201-invariance-and-stability-of-deep-convolutional-representations)"
  - theory paper

## wavelet style transfer

- Photorealistic Style Transfer via Wavelet Transforms> ([yoo et al. 2019](https://arxiv.org/pdf/1903.09760v1.pdf))

# adaptive wavelet papers

- Parameterized Wavelets for Convolutional Neural Networks ([2020](https://ieeexplore-ieee-org.libproxy.berkeley.edu/stamp/stamp.jsp?tp=&arnumber=9096125&tag=1)) - a discrete wavelet CNN
- An End-to-End Multi-Level Wavelet Convolutional Neural Networks for heart diseases diagnosis ([el bouny et al. 2020](https://www.sciencedirect.com/science/article/abs/pii/S0925231220311644)) - stationary wavelet CNN
- Fully Learnable Deep Wavelet Transform for Unsupervised Monitoring of High-Frequency Time Series ([michau et al. 2021](https://arxiv.org/abs/2105.00899))

## helmut lab papers

- Deep Convolutional Neural Networks Based on Semi-Discrete Frames ([wiatowski et al. 2015](https://arxiv.org/abs/1504.05487))
  - allowing for different and, most importantly, general semidiscrete frames (such as, e.g., Gabor frames, wavelets, curvelets, shearlets, ridgelets) in distinct network layers
  - translation-invariant, and we develop deformation stability results
- wiatoski_18 "[A mathematical theory of deep convolutional neural networks for feature extraction](https://ieeexplore.ieee.org/document/8116648/)"
  - encompasses general convolutional transforms - general semi-discrete frames (including Weyl-Heisenberg filters, curvelets, shearlets, ridgelets, wavelets, and learned filters), general Lipschitz-continuous non-linearities (e.g., rectified linear units, shifted logistic sigmoids, hyperbolic tangents, and modulus functions), and general Lipschitz-continuous pooling operators emulating, e.g., sub-sampling and averaging
    - all of these elements can be different in different network layers.
  - translation invariance result of vertical nature in the sense of the features becoming progressively more translation-invariant with increasing network depth
  - deformation sensitivity bounds that apply to signal classes such as, e.g., band-limited functions, cartoon functions, and Lipschitz functions.
- wiatowski_18 "Energy Propagation in Deep Convolutional Neural Networks" ([wiatowski et al. 2017](https://arxiv.org/pdf/1704.03636.pdf))

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

## l2 functions

- $L^2$ function is a function $f: X \to \mathbb{R}$ that is square integrable: $$\vert f\vert ^2 = \int_X \vert f\vert ^2 d\mu$$ with respect to the measure $\mu$
  - $$\vert f\vert $$ is its $L^2$-norm
- **measure** = nonnegative real function from a delta-ring F such that $m(\empty) = 0$ and $m(A) = \sum_n m(A_n)$
- **Hilbert space** H: a vectors space with an innor product $$<f, g>$$ such that the following norm turns H into a complete metric space: $$\vert f\vert  = \sqrt{<f, f>}$$
- **diffeomorphism** is an [isomorphism](https://en.wikipedia.org/wiki/Isomorphism) of [smooth manifolds](https://en.wikipedia.org/wiki/Smooth_manifold). It is an [invertible function](https://en.wikipedia.org/wiki/Invertible_function) that [maps](https://en.wikipedia.org/wiki/Map_(mathematics)) one [differentiable manifold](https://en.wikipedia.org/wiki/Differentiable_manifold) to another such that both the function and its inverse are [smooth](https://en.wikipedia.org/wiki/Smooth_function).


## reversible/invertible models

- williams  inverting w/ autoregressive models ([nash et al. 2018](https://arxiv.org/abs/1806.00400))
- [arora reversibility](https://arxiv.org/pdf/1511.05653.pdf)
- iResNet
