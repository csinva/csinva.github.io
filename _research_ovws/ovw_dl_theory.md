---
layout: notes_without_title
section-type: notes
title: dl theory
category: research
---

*Deep learning theory is a complex emerging field - this post contains links displaying some different interesting research directions*

[TOC]

# theoretical methods

DNNs display many surprising properties

- surprising: [more parameters yields better generalization](https://arxiv.org/abs/1802.08760) 
- surprising: lowering training error should be harder

Many things seem to contribute to the inductive bias of DNNs: SGD, dropout, early stopping, resnets, convolution, more layers...all of these are tangled together and many things correlate with generalization error...what are the important things and how do they contribute? 

some more concrete questions:

- what is happening when training err stops going down but val err keeps going down (interpolation regime)?
- what are good statistical markers of an effectively trained DNN?
- how far apart are 2 nets?

## functional approximation

- nonlinear approximation (e.g. sparse coding) - 2 steps

  1. construct a dictionary function (T)
  2. learn linear combination of the dictionary elements (g)
- background
  - $L^2$ function (or function space) is square integrable: $|f|^2 = \int_X |f|^2 d\mu$, and $|f|$ is its $L_2$-norm
  - **Hilbert space** - vector space w/ additional structure of inner product which allows length + angle to be measured
    - complete - there are enough limits in the space to allow calculus techniques (is a *complete metric space*)
- composition allows an approximation of a function through level sets (split it up and approximate on these sets) - Zuowei Shen ([talk](http://www.ipam.ucla.edu/programs/workshops/workshop-iii-geometry-of-big-data/?tab=schedule), [slides](../../../drive/papers/dl_theory/shen_19_composition_dl_slides.pdf)) 
  - composition operation allows an approximation of a function f through level sets of f --
  - one divides up the range of f into equal intervals and approximate the functions on these sets 
  - [nonlinear approximation via compositions](https://arxiv.org/pdf/1902.10170.pdf) (shen 2019)
- how do the weights in each layer help this approximation to be more effective?
  - here are some thoughts -- If other layers are like the first layer, the weights "whiten" or make the inputs more independent or random projections -- that is basically finding PC directions for low-rank inputs. 
  - Are the outputs from later layers more or less low - rank?
  - I wonder how this "whitening" helps level set estimation...

## inductive bias: gradient descent finds good minima

*DL learns solutions that generalize even though it can find many which don't due to its inductive bias.*

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
- [implicit bias towards simpler models](https://arxiv.org/abs/1805.08522)
- [How do infinite width bounded norm networks look in function space?](https://arxiv.org/pdf/1902.05040.pdf) (savarese...srebro 2019)
  - minimal norm fit for a sample is given by a linear spline interpolation (2 layer net)
- [analytic theory of generalization + transfer (ganguli 19)](https://arxiv.org/abs/1809.10374)
  - deep linear nets learn important structure of the data first (less noisy eigenvectors)
  - [similar paper for layer nets](https://arxiv.org/pdf/1904.13262.pdf)
- datasets for measuring causality
  - Inferring Hidden Statuses and Actions in Video by Causal Reasoning - about finding causality in the video, not interpretation

## semantic biases: what correlations will a net learn?

- [imagenet models are biased towards texture](https://arxiv.org/abs/1811.12231) (and removing texture makes them more robust)
- [analyzing semantic robustness](https://arxiv.org/pdf/1904.04621.pdf)
- [eval w/ simulations](https://arxiv.org/abs/1712.06302) (reviewer argued against this)
- [glcm captures superficial statistics](https://arxiv.org/abs/1903.06256)
- [deeper, unpruned networks are better against noise](https://arxiv.org/abs/1903.12261)
- [analytic theory of generalization + transfer (ganguli 19)](https://arxiv.org/abs/1809.10374)
- [causality in dnns talk by bottou](https://www.technologyreview.com/s/613502/deep-learning-could-reveal-why-the-world-works-the-way-it-does/)
  - on mnist, color vs shape will learn color
- [A mathematical theory of semantic development in deep neural networks](https://arxiv.org/abs/1810.10531)

## expressiveness: what can a dnn repreresent?

- [complexity of linear regions in deep networks](https://arxiv.org/pdf/1901.09021.pdf)
- [Bounding and Counting Linear Regions of Deep Neural Networks](https://arxiv.org/pdf/1711.02114.pdf)
- [On the Expressive Power of Deep Neural Networks](https://arxiv.org/pdf/1606.05336.pdf)



## complexity + generalization: dnns are low-rank / redundant parameters

- measuring complexity
  - [functional decomposition](https://arxiv.org/abs/1904.03867) (molnar 2019)
    - decompose function into bias + first-order effects (ALE) + interactions
    - 3 things: number of features used, interaction strength, main effect complexity
- parameters are redundant
  - [predicting params](http://papers.nips.cc/paper/5025-predicting-parameters-in-deep-learning): weight matrices are low-rank, decompose into UV by picking a U
  - [pruning neurons](https://arxiv.org/abs/1507.06149)
  - [circulant projection](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Cheng_An_Exploration_of_ICCV_2015_paper.pdf)
  - [rethinking the value of pruning](https://arxiv.org/pdf/1810.05270.pdf): pruning and training from scratch, upto 30% size
  - [Lottery ticket](https://openreview.net/pdf?id=rJl-b3RcF7): pruning and training from initial random weights, upto 1% size ([followup](https://arxiv.org/abs/1903.01611))
  - [rank of relu activations](https://arxiv.org/pdf/1810.03372.pdf)
  - [random weights are good](https://arxiv.org/pdf/1504.08291.pdf)
  - [singular values of conv layers](https://arxiv.org/pdf/1805.10408.pdf)
  - [T-Net: Parametrizing Fully Convolutional Nets with a Single High-Order Tensor](https://arxiv.org/abs/1904.02698)
- generalization
  - [size of the weights is more important](http://eprints.qut.edu.au/43927/)
  - [Quantifying the generalization error in deep learning in terms of data distribution and
    neural network smoothness](https://arxiv.org/pdf/1905.11427v1.pdf)

## double descent: bias + variance in high dims?

- [Reconciling modern machine learning and the bias-variance trade-off](https://arxiv.org/abs/1812.11118) (belkin et al. 2018)
- [Surprises in High-Dimensional Ridgeless Least Squares Interpolation](https://arxiv.org/abs/1903.08560)
  - main result of limiting risk, where $\gamma \in (0, \infty)$:
    - $R(\gamma) = \begin{cases} \sigma^2 \frac{\gamma}{1-\gamma} & \gamma < 1\\||\beta||_2^2(1 - \frac 1 \gamma) + \sigma^2 \frac{1} {\gamma - 1} & \gamma > 1\end{cases}$
- [linear regression depends on data distr.](https://arxiv.org/abs/1802.05801)
- [two models of double descent for weak features](https://arxiv.org/abs/1903.07571)
- [double descent curve](https://openreview.net/forum?id=HkgmzhC5F7)
- [boosting w/ l2 loss](https://www.tandfonline.com/doi/pdf/10.1198/016214503000125?casa_token=5OE5LZe_mIcAAAAA:-4DdXLa4A6SeXnguyYv1S3bfIbRXrSb1qojj_UkGZpmbNHqjkWMojm0al5xx2yz-7ABcfDXmdvBeCw)
- [effective degrees of freedom](https://web.stanford.edu/~hastie/Papers/df_paper_LJrev6.pdf)
- [high-dimensional ridge](https://projecteuclid.org/euclid.aos/1519268430)

## kernels

- [kernels wiki](https://en.wikipedia.org/wiki/Kernel_method#cite_note-4): kernel memorizes points then uses dists between points to classify
- [learning deep kernels](https://arxiv.org/pdf/1811.08357v1.pdf)
- [learning data-adaptive kernels](https://arxiv.org/abs/1901.07114)
- [kernels that mimic dl](https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf)
- [kernel methods](http://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdfs)
- [neural tangent kernel](https://arxiv.org/abs/1806.07572) (jacot et al. 2018)
  - at initialization, artificial neural networks (ANNs) are equivalent to Gaussian
    processes in the infinite-width limit
    - evolution of an ANN during training can also be described by a kernel (kernel gradient descent)
- [Scaling description of generalization with number of parameters in deep learning](https://arxiv.org/abs/1901.01608) (geiger et al. 2019)
  - number of params = N
  - above 0 training err, larger number of params reduces variance but doesn't actually help
    - ensembling with smaller N fixes problem
  - the improvement of generalization performance with N in this classification task originates from reduced variance of fN when N gets large, as recently observed for mean-square regression
- [understanding the neural tangent kernel](https://arxiv.org/pdf/1904.11955.pdf) (arora et al. 2019)
  - method to compute the kernel quickly on a gpu

## nearest neighbor comparisons

- [weighted interpolating nearest neighbors can generalize well (belkin...mitra 2018)](http://papers.nips.cc/paper/7498-overfitting-or-perfect-fitting-risk-bounds-for-classification-and-regression-rules-that-interpolate)
- [Statistical Optimality of Interpolated Nearest Neighbor Algorithms](https://arxiv.org/abs/1810.02814)
- [nearest neighbor comparison](https://arxiv.org/pdf/1805.06822.pdf)
- [nearest embedding neighbors](https://arxiv.org/pdf/1803.04765.pdf)

## random projections

- [sgd for polynomials](http://proceedings.mlr.press/v32/andoni14.pdf)
- [deep linear better than just linear](https://arxiv.org/pdf/1811.10495.pdf)
- relation to bousquet - fitting random polynomials
- [hierarchical sparse coding for images](https://pdfs.semanticscholar.org/9636/d8aedd476ef19c762923119750aec95bf8ca.pdf) (can’t just repeat sparse coding, need to include input again)
- [random projections in the brain](https://www.biorxiv.org/content/biorxiv/early/2017/08/25/180471.full.pdf)….doing locality sensitive hashing (basically nearest neighbors)

## robustness

- [robustness may be at odds with accuracy](https://openreview.net/pdf?id=SyxAb30cY7) (madry 2019)
  - adversarial training helps w/ little data but hurts with lots of data
  - adversarially trained models have more meaningful gradients (and their adversarial examples actually look like other classes)
- [Generalizability vs. Robustness: Adversarial Examples for Medical Imaging](https://arxiv.org/abs/1804.00504)
- [robustness of explanations](https://arxiv.org/pdf/1806.07538.pdf)

# empirical studies

##  interesting empirical papers

- [modularity (“lottery ticket hypothesis”)](https://arxiv.org/abs/1803.03635) 
  - contemporary experience is that it is difficult to train small architectures from scratch, which would similarly improve training performance - **lottery ticket hypothesis**: large networks that train successfully contain subnetworks that--when trained in isolation--converge in a comparable number of iterations to comparable accuracy
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

## comparing representations

- [shared representations across nets](https://arxiv.org/abs/1811.11684)
- [comparing across random initializations](https://arxiv.org/abs/1810.11750)



## simple papers

- [rvw of random features approach](https://arxiv.org/pdf/1904.00687.pdf)
- [similar nets learn different weights](http://proceedings.mlr.press/v44/li15convergent.pdf)



## adam vs sgd papers

- svd parameterization rnn paper: [inderjit paper](https://arxiv.org/pdf/1803.09327.pdf)
    - original adam paper: [kingma 15](https://arxiv.org/abs/1412.6980)
    - [regularization via SGD](https://arxiv.org/pdf/1806.00900.pdf) (layers are balanced: du 18)
    - marginal value of adaptive methods: [recht 17](http://papers.nips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning)
    - comparing representations: [svcca](https://arxiv.org/abs/1706.05806)
    - [montanari 18](https://arxiv.org/pdf/1804.06561.pdf) pde mean field view
    - [normalized margin bounds](http://papers.nips.cc/paper/7204-spectrally-normalized-margin-bounds-for-neural-networks)

## memorization background

- [memorization on single training example](https://arxiv.org/abs/1902.04698v2)
- [memorization in dnns](https://arxiv.org/pdf/1706.05394.pdf)
  
- “memorization” as the behavior exhibited by DNNs trained on noise, and conduct a series of experiments that contrast the learning dynamics of DNNs on real vs. noise data
  - look at critical samples - adversarial exists nearby
  
- networks that generalize well have deep layers that are approximately linear with respect to batches of similar inputs
  - networks that memorize their training data are highly non-linear with respect to similar inputs, even in deep layers
  - expect that with respect to a single class, deep layers are approximately linear
  
- [example forgetting paper](https://openreview.net/forum?id=BJlxm30cKm)
  
- [secret sharing](https://arxiv.org/abs/1802.08232)

## probabilistic inference

- multilayer idea
- [equilibrium propagation](https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full)

## architecture search background

- ideas: nested search (retrain each time), joint search (make arch search differentiable), one-shot search (train big net then search for subnet)
  - [asap: online pruning + training](https://arxiv.org/pdf/1904.04123.pdf)
  - [rvw](https://arxiv.org/abs/1904.00438)
  - [original (dumb) strategy](https://arxiv.org/abs/1611.01578)
  - [progressive nas](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf)
  - [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268) - sharing params
  - [randomly replace layers with the identity when training](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39)
  - [learning both weights and connections](https://pdfs.semanticscholar.org/1ff9/a37d766e3a4f39757f5e1b235a42dacf18ff.pdf)
  - [single-path one-shot search](https://arxiv.org/abs/1904.00420)


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