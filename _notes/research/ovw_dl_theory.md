---
layout: notes_without_title
section-type: notes
title: dl theory primer
category: research
---

*Deep learning theory is a complex emerging field - this post contains links displaying some different new research directions*

[TOC]

# general research ovw
DNNs display many surprising properties

- surprising: [more parameters yields better generalization](https://arxiv.org/abs/1802.08760) 
- surprising: lowering training error should be harder

Many things seem to contribute to the inductive bias of DNNs: SGD, dropout, early stopping, resnets, convolution, more layers...all of these are tangled together and many things correlate with generalization error...what are the important things and how do they contribute? 

Some more concrete questions are

- what is happening when training err stops going down but val err keeps going down (interpolation regime)?
- what are good statistical markers of an effectively trained DNN?
- how far apart are 2 nets?


## gradient descent finds good minima

- [gd bousquet paper](https://arxiv.org/pdf/1803.08367.pdf) 
- [in high dims, local minima are usually saddles (ganguli)](http://papers.nips.cc/paper/5486-identifying-and-attacking-the-saddle-point-problem-in-high-dimensional-non-convex-optimization)
- [srebro understanding over-parameterization	](https://arxiv.org/abs/1805.12076) 
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

## generalization

- generalization
  - [size of the weights is more important](http://eprints.qut.edu.au/43927/)
  - [Reconciling modern machine learning and the bias-variance trade-off](https://arxiv.org/abs/1812.11118)
  - Nati Srebro papers

## kernels

- [kernels](https://en.wikipedia.org/wiki/Kernel_method#cite_note-4): kernel memorizes points then uses dists between points to classify
- [learning deep kernels](https://arxiv.org/pdf/1811.08357v1.pdf)
- [learning data-adaptive kernels](https://arxiv.org/abs/1901.07114)
- [kernels that mimic dl](https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf)
- [kernel methods](http://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdfs)


## nearest neighbor comparisons
- [weighted interpolating nearest neighbors can generalize well (belkin...mitra 2018)](http://papers.nips.cc/paper/7498-overfitting-or-perfect-fitting-risk-bounds-for-classification-and-regression-rules-that-interpolate)
- [nearest neighbor comparison](https://arxiv.org/pdf/1805.06822.pdf)
- [nearest embedding neighbors](https://arxiv.org/pdf/1803.04765.pdf)

## random projections

- [sgd for polynomials](http://proceedings.mlr.press/v32/andoni14.pdf)
- [deep linear better than just linear](https://arxiv.org/pdf/1811.10495.pdf)
- relation to bousquet - fitting random polynomials
- [hierarchical sparse coding for images](https://pdfs.semanticscholar.org/9636/d8aedd476ef19c762923119750aec95bf8ca.pdf) (can’t just repeat sparse coding, need to include input again)
- [random projections in the brain](https://www.biorxiv.org/content/biorxiv/early/2017/08/25/180471.full.pdf)….doing locality sensitive hashing (basically nearest neighbors)

##  interesting empirical papers

- [modularity (“lottery ticket hypothesis”)](https://arxiv.org/abs/1803.03635) 
- contemporary experience is that it is difficult to train small architectures from scratch, which would similarly improve training performance - **lottery ticket hypothesis**: It states that large networks that train successfully contain subnetworks that--when trained in isolation--converge in a comparable number of iterations to comparable accuracy
- [ablation studies](https://arxiv.org/abs/1812.05687)
- [deep learning is robust to massive label noise](https://arxiv.org/pdf/1705.10694.pdf)

## tools for analyzing

- dim reduction: [svcca](http://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-understanding-and-improvement), diffusion maps
- viz tools: [bei wang](http://www.sci.utah.edu/~beiwang/)
- [visualizing loss landscape](https://arxiv.org/pdf/1712.09913.pdf)
- 1d: plot loss by extrapolating between 2 points (start/end, 2 ends)
  - goodfellow et al. 2015, im et al. 2016
  - [exploring landscape with this technique](https://arxiv.org/pdf/1803.00885.pdf)
  - 2d: plot loss on a grid in 2 directions
  - important to think about scale invariance (dinh et al. 2017)
  - want to scale direction vector to have same norm in each direction as filter
  - use PCA to find important directions (ex. sample w at each step, pca to find most important directions of variance)

## misc theoretical areas
- deep vs. shallow [rvw](http://cbmm.mit.edu/sites/default/files/publications/CBMM-Memo-058v5.pdf)
- [probabilistic framework](https://www.nari.ee.ethz.ch/commth//pubs/files/deep-2016.pdf)
- information bottleneck: tishby paper + david cox follow-up
- [manifold learning](https://www.deeplearningbook.org/version-2015-10-03/contents/manifolds.html)
  - [random manifold learning paper](https://ieeexplore.ieee.org/document/7348689/) 

# basics

- [good set of class notes](https://people.csail.mit.edu/madry/6.883/)
- demos to gain intuition
  - [colah](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) 
    - [tf playground](https://playground.tensorflow.org/)
    - [convnetJS](https://cs.stanford.edu/people/karpathy/convnetjs//demo/classify2d.html)
    - [ml playground](http://ml-playground.com/)

  - overview / reviews

    - [mathematics of dl](https://arxiv.org/abs/1712.04741)
    - [stanford class](https://stats385.github.io/) ([good readings](https://stats385.github.io/readings))
    - [dali 2018 talks](http://dalimeeting.org/dali2018/workshopTheoryDL.html)
    - [overview (myths)](http://www.mit.edu/~rakhlin/papers/myths.pdf) 
- some people involved: nathan srebro, sanjeev arora, jascha sohl-dickstein, tomaso poggio, stefano soatto, ben recht, [olivier bousquet](https://arxiv.org/abs/1803.08367), jason lee, simon shaolei du
  - interpretability: cynthia rudin, rich caruana, been kim, nicholas papernot, finale doshi-velez
  - neuro: eero simoncelli, haim sompolinsky