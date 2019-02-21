---
layout: notes_without_title
section-type: notes
title: dl theory ref
category: research
---

*Deep learning theory is a complex emerging field - this post contains links displaying some different new research directions*

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
    - arora paper: things correlate w/ generalization error

  - properties	

    - architectures merge model with optimization
    - surprising: [more parameters yields better generalization](https://arxiv.org/abs/1802.08760) 
    - surprising: lowering training error should be harder

# general research ovw
- some people involved
  - theory: jascha sohl-dickstein, tomaso poggio, stefano soatto, ben recht, sanjeev arora, [olivier bousquet](https://arxiv.org/abs/1803.08367), nathan srebro; Simon Shaolei Du
  - interpretability: cynthia rudin, rich caruana, been kim, nicholas papernot, finale doshi-velez

- [manifold learning](https://www.deeplearningbook.org/version-2015-10-03/contents/manifolds.html)

- [random manifold learning paper](https://ieeexplore.ieee.org/document/7348689/) 

  - dim reduction: [svcca](http://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-understanding-and-improvement), diffusion maps

- [kernels](https://en.wikipedia.org/wiki/Kernel_method#cite_note-4): kernel memorizes points then uses dists between points to classify

- - [learning deep kernels](https://arxiv.org/pdf/1811.08357v1.pdf)
  - [learning data-adaptive kernels](https://arxiv.org/abs/1901.07114)
  - [kernels that mimic dl](https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf)

- architectures

- [residual connections?](https://arxiv.org/pdf/1611.01186.pdf)

  - deep vs. shallow [rvw](http://cbmm.mit.edu/sites/default/files/publications/CBMM-Memo-058v5.pdf)

  - gaussian processes: outputs uncertainty

  - [probabilistic framework](https://www.nari.ee.ethz.ch/commth//pubs/files/deep-2016.pdf)

  - invertible models

  - [williams  inverting w/ autoregressive models](https://arxiv.org/abs/1806.00400)
    - [arora reversibility](https://arxiv.org/pdf/1511.05653.pdf) 

  - [characterizing CNNs explicitly with transforms](https://www.nari.ee.ethz.ch/commth//pubs/files/deep-2016.pdf)

  - [modularity (“lottery ticket hypothesis”)](https://arxiv.org/abs/1803.03635) 

  - contemporary experience is that it is difficult to train small architectures from scratch, which would similarly improve training performance. 
    - "lottery ticket hypothesis": It states that large networks that train successfully contain subnetworks that--when trained in isolation--converge in a comparable number of iterations to comparable accuracy

- sparse coding

- elms

  - [sgd for polynomials](http://proceedings.mlr.press/v32/andoni14.pdf)
  - [deep linear better than just linear](https://arxiv.org/pdf/1811.10495.pdf)
  - relation to bousquet - fitting random polynomials
  - [hierarchical sparse coding for images](https://pdfs.semanticscholar.org/9636/d8aedd476ef19c762923119750aec95bf8ca.pdf) (can’t just repeat sparse coding, need to include input again)
  - [random projections in the brain](https://www.biorxiv.org/content/biorxiv/early/2017/08/25/180471.full.pdf)….doing locality sensitive hashing (basically nearest neighbors)

- discussion with rajiv

- visualize with [bei wang](http://www.sci.utah.edu/~beiwang/)

  - resnet blocks w/ boosting: <https://arxiv.org/pdf/1706.04964.pdf> 
  - l1-regularized nns: <http://proceedings.mlr.press/v48/zhangd16.pdf> 

- regularization

- initialization

  - GD/sgd

  - discreteness of GD
    - SGD noise
    - [gd bousquet paper](https://arxiv.org/pdf/1803.08367.pdf) 
    - [separable gd bias](https://openreview.net/pdf?id=r1q7n9gAb)

  - dropout

  - early stopping

- optimization: local flat minima?

- information bottleneck: tishby paper + david cox follow-up

  - [kernel methods](http://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdfs)

  - [nearest neighbor comparison](https://arxiv.org/pdf/1805.06822.pdf)

  - [nearest embedding neighbors](https://arxiv.org/pdf/1803.04765.pdf)

  - [srebro understanding over-parameterization	](https://arxiv.org/abs/1805.12076)

- analyze the loss surface

- generalization gap?

  - how many local minima?

  - [visualizing loss landscape](https://arxiv.org/pdf/1712.09913.pdf)

  - 1d: plot loss by extrapolating between 2 points (start/end, 2 ends)
    - goodfellow et al. 2015
      - im et al. 2016
      - [exploring landscape with this technique](https://arxiv.org/pdf/1803.00885.pdf)

    - 2d: plot loss on a grid in 2 directions

    - important to think about scale invariance (dinh et al. 2017)
      - want to scale direction vector to have same norm in each direction as filter
      - random directions or ok
      - use PCA to find important directions (ex. sample w at each step, pca to find most important directions of variance)

- connecting with neuroscience

- - [ablation studies](https://arxiv.org/abs/1812.05687)