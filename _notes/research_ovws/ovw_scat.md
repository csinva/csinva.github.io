---
layout: notes
title: scattering transform
category: research
---

{:toc}

**some papers involving the scattering transform and similar developments bringing structure to replace learned filters**

- some of the researchers involved
  - edouard oyallan, joan bruna, stephan mallat, Helmut Bölcskei, max welling

# goals

- benefits        
   - all filters are defined
   - more interpretable
   - more biophysically plausible
- scattering transform - computes a translation invariant repr. by cascading wavelet transforms and modulus pooling operators, which average the amplitude of iterated wavelet coefficients

# review-type
- [Understanding deep convolutional networks](https://arxiv.org/abs/1601.04920) (mallat 2016)
- [Mathematics of deep learning](https://arxiv.org/abs/1712.04741) (vidal et al. 2017)
- [Geometric deep learning: going beyond euclidean data](https://arxiv.org/abs/1611.08097) (bronstein et al. 2017)

# initial papers

- [classification with scattering operators](https://arxiv.org/abs/1011.3023) (bruna & mallat 2010)
- [recursive interferometric repr.](https://www.di.ens.fr/data/publications/papers/Eusipco2010InterConfPap.pdf) (mallat 2010)
- [group invariant scattering](https://arxiv.org/abs/1101.2286) (mallat 2012)
  - introduces scat transform
- [Generic deep networks with wavelet scattering](https://arxiv.org/abs/1312.5940) (oyallan et al. 2013)
- [Invariant Scattering Convolution Networks](https://arxiv.org/abs/1203.1513) (bruna & mallat 2012)
   - introduces the scattering transform implemented as a cnn
- [Deep scattering spectrum](https://arxiv.org/abs/1304.6763) (anden & mallat 2013)

## scat_conv

- [Deep roto-translation scattering for object classification](https://arxiv.org/abs/1412.8659) (oyallan & mallat 2014)
    - use 1x1 conv on top of scattering coefs (only 1 layer)
    - can capture rounded figures
    - can further impose robustness to rotation variability (although not full rotation invariance)
    - [Deep learning in the wavelet domain](https://arxiv.org/pdf/1811.06115.pdf) (cotter & kingbury, 2017)- each conv layer is replaced by scattering transform + 1x1 conv
- [Visualizing and improving scattering networks](https://arxiv.org/pdf/1709.01355.pdf) (cotter et al. 2017)
  - add deconvnet to visualize
- [Scattering Networks for Hybrid Representation Learning](https://hal.inria.fr/hal-01837587/document) (oyallon et al. 2018)
    - using early layers scat is good enough
- [i-RevNet: Deep Invertible Networks](https://arxiv.org/abs/1802.07088) (jacobsen et al. 2018)
- [Scaling the scattering transform: Deep hybrid networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Oyallon_Scaling_the_Scattering_ICCV_2017_paper.pdf) (oyallon et al. 2017)
    - use 1x1 convolutions to collapse accross channels
- jacobsen_17 "Hierarchical Attribute CNNs"
    - modularity
- cheng_16 "Deep Haar scattering networks"
- [Deep Network Classification by Scattering and Homotopy Dictionary Learning](https://arxiv.org/abs/1910.03561) (zarka et al. 2019) - scat followed by sparse coding then linear

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

- [Photorealistic Style Transfer via Wavelet Transforms>](https://arxiv.org/pdf/1903.09760v1.pdf)

# adaptive wavelet papers

- [Parameterized Wavelets for Convolutional Neural Networks](https://ieeexplore-ieee-org.libproxy.berkeley.edu/stamp/stamp.jsp?tp=&arnumber=9096125&tag=1) (2020) - a discrete wavelet CNN
- [An End-to-End Multi-Level Wavelet Convolutional Neural Networks for heart diseases diagnosis ](https://www.sciencedirect.com/science/article/abs/pii/S0925231220311644) (el bouny et al. 2020) - stationary wavelet CNN
- [Fully Learnable Deep Wavelet Transform for Unsupervised Monitoring of High-Frequency Time Series](https://arxiv.org/abs/2105.00899) (michau et al. 2021)

## helmut lab papers

- [Deep Convolutional Neural Networks Based on Semi-Discrete Frames](https://arxiv.org/abs/1504.05487) (wiatowski et al. 2015)
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

## l2 functions

- $L^2$ function is a function $f: X \to \mathbb{R}$ that is square integrable: $|f|^2 = \int_X |f|^2 d\mu$ with respect to the measure $\mu$
  - $|f|$ is its $L^2$-norm
- **measure ** = nonnegative real function from a delta-ring F such that $m(\empty) = 0$ and $m(A) = \sum_n m(A_n)$
- **Hilbert space** H: a vectors space with an innor product $<f, g>$ such that the following norm turns H into a complete metric space: $|f| = \sqrt{<f, f>}$
- **diffeomorphism** is an [isomorphism](https://en.wikipedia.org/wiki/Isomorphism) of [smooth manifolds](https://en.wikipedia.org/wiki/Smooth_manifold). It is an [invertible function](https://en.wikipedia.org/wiki/Invertible_function) that [maps](https://en.wikipedia.org/wiki/Map_(mathematics)) one [differentiable manifold](https://en.wikipedia.org/wiki/Differentiable_manifold) to another such that both the function and its inverse are [smooth](https://en.wikipedia.org/wiki/Smooth_function).


## reversible/invertible models

- [williams  inverting w/ autoregressive models](https://arxiv.org/abs/1806.00400)

- [arora reversibility](https://arxiv.org/pdf/1511.05653.pdf)
- iResNet
