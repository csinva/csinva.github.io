---
layout: notes_without_title
section-type: notes
title: interp
category: research
---

**some papers I like involving interpretable machine learning, references from this [interpretable ml review](https://arxiv.org/abs/1901.04592), and some notes from the [interpretable ml book](https://christophm.github.io/interpretable-ml-book/)**

[TOC]

# misc new papers

- [Beyond Sparsity: Tree Regularization of Deep Models for Interpretability](https://arxiv.org/pdf/1711.06178.pdf)
  - regularize so that deep model can be closely modeled by tree w/ few nodes
- [THE CONVOLUTIONAL TSETLIN MACHINE](https://arxiv.org/pdf/1905.09688.pdf)
  - [The Tsetlin Machine](https://arxiv.org/pdf/1804.01508.pdf)
- [explaining image classifiers by counterfactual generation](https://arxiv.org/pdf/1807.08024.pdf) 
  - generate changes (e.g. with GAN in-filling) and see if pred actually changes
- [ConvNets and ImageNet Beyond Accuracy: Understanding Mistakes and Uncovering Biases](https://arxiv.org/abs/1711.11443)
  - cnns are more accurate, robust, and biased then we might expect on imagenet
- [BRIDGING ADVERSARIAL ROBUSTNESS AND GRADIENT INTERPRETABILITY](https://arxiv.org/abs/1903.11626)
- [Harnessing Deep Neural Networks with Logic Rules](https://arxiv.org/pdf/1603.06318.pdf)
- Exploring Principled Visualizations for Deep Network Attributions - viz of attribution can be misleading (might want to clip, etc.)
  - layer separation - want to have both image and attributions on top of it 
- [explaining a black-box w/ deep variational bottleneck](https://arxiv.org/abs/1902.06918)
- [discovering and testing visual concepts learned by DNNs](https://arxiv.org/abs/1902.03129) - cluster in bottleneck space
- [interpretable filters](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0490.pdf)
- [global explanations](https://arxiv.org/abs/1902.02384)
- [bagnet bag-of-features](https://openreview.net/pdf?id=SkfMWhAqYQ)
  - [learn shapes not texture](https://openreview.net/pdf?id=Bygh9j09KX)
  - [code](https://github.com/wielandbrendel/bag-of-local-features-models)
- [neural stethoscopes](https://arxiv.org/pdf/1806.05502.pdf) 
- [RISE](https://arxiv.org/pdf/1806.07421.pdf) - randomized input sampling
- [xGEMs](https://arxiv.org/pdf/1806.08867.pdf) 
- [maximally invariant data perturbation](https://arxiv.org/pdf/1806.07004.pdf)
- hard coding
  - [SSIM layer](https://arxiv.org/abs/1806.09152)
  - Inverting Supervised Representations with Autoregressive Neural Density Models 
- robustness
  - https://arxiv.org/pdf/1806.08049.pdf
  - https://arxiv.org/pdf/1806.07538.pdf
- [piecewise linear interp](https://arxiv.org/pdf/1806.10270.pdf)
- [nonparametric var importance](http://proceedings.mlr.press/v80/feng18a/feng18a.pdf)
- [supervised local modeling](https://arxiv.org/abs/1807.02910 ) 
- [detect adversarial cnn attacks w/ feature maps](https://digitalcollection.zhaw.ch/handle/11475/8027) 
- [adaptive dropout](https://arxiv.org/abs/1807.08024)
- [lesion detection saliency](https://arxiv.org/pdf/1807.07784.pdf) 
- [integrated gradients 2](https://arxiv.org/abs/1805.12233)
- [symbolic execution for dnns](https://arxiv.org/pdf/1807.10439.pdf)
- [L-shapley abd C-shapley](https://arxiv.org/pdf/1808.02610.pdf)
- [Understanding Deep Architectures by Visual Summaries](http://bmvc2018.org/papers/0794.pdf)
- [A Simple and Effective Model-Based Variable Importance Measure](https://arxiv.org/pdf/1805.04755.pdf)
  - measures the feature importance (defined as the variance of the 1D partial dependence function) of one feature conditional on different, fixed points of the other feature. When the variance is high, then the features interact with each other, if it is zero, they don’t interact.
- random forests
  - Breiman proposes permutation tests: Breiman, Leo. 2001. “Random Forests.” Machine Learning 45 (1). Springer: 5–32
- [Interpreting Neural Network Judgments via Minimal, Stable, and Symbolic Corrections](https://arxiv.org/pdf/1802.07384.pdf)
- [DeepPINK: reproducible feature selection in deep neural networks](https://arxiv.org/pdf/1809.01185.pdf)
- "Transparency by Disentangling Interactions"
- "To Trust Or Not To Trust A Classifier"
- "Interpreting Neural Network Judgments via Minimal, Stable, and Symbolic Corrections"
- [Towards Robust Interpretability with Self-Explaining Neural Networks](https://arxiv.org/pdf/1806.07538.pdf)
- "Explaining Deep Learning Models -- A Bayesian Non-parametric Approach"
- [Detecting Potential Local Adversarial Examples for Human-Interpretable Defense](https://arxiv.org/pdf/1809.02397.pdf)
- [Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning](https://arxiv.org/pdf/1803.04765.pdf)
  - [Interpreting Neural Networks With Nearest Neighbors](https://arxiv.org/pdf/1809.02847.pdf)
- [Generalizability vs. Robustness: Adversarial Examples for Medical Imaging](https://arxiv.org/abs/1804.00504)
- [Interpreting Layered Neural Networks via Hierarchical Modular Representation](https://arxiv.org/pdf/1810.01588.pdf)
- [Entropic Variable Boosting for Explainability & Interpretability in Machine Learning](https://arxiv.org/abs/1810.07924)
- [Explain to Fix: A Framework to Interpret and Correct DNN Object Detector Predictions](https://arxiv.org/pdf/1811.08011.pdf)
- [Understanding Individual Decisions of CNNs via Contrastive Backpropagation](https://arxiv.org/abs/1812.02100v1)
- [“What are You Listening to?” Explaining Predictions of Deep Machine Listening Systems](https://ieeexplore.ieee.org/abstract/document/8553178)
- [Diagnostic Visualization for Deep Neural Networks Using Stochastic Gradient Langevin Dynamics](https://arxiv.org/pdf/1812.04604.pdf)
- [Interpretable Convolutional Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0490.pdf)
  - could also just use specific filters for specific classes
- [Manipulating and Measuring Model Interpretability](https://arxiv.org/abs/1802.07810)
- local
  - [grad-cam++](https://arxiv.org/abs/1710.11063)
- importance scores
  - [Variable Importance Clouds: A Way to Explore Variable Importance for the Set of Good Models](https://arxiv.org/pdf/1901.03209.pdf) 
  - [Permutation tests in general](http://arxiv.org/abs/1801.01489): Fisher, Aaron, Cynthia Rudin, and Francesca Dominici. 2018. “Model Class Reliance: Variable Importance Measures for any Machine Learning Model Class, from the ‘Rashomon’ Perspective.”
- [Recovering Pairwise Interactions Using Neural Networks](https://arxiv.org/pdf/1901.08361.pdf)
- [How Sensitive are Sensitivity-Based Explanations?](https://arxiv.org/abs/1901.09392)
- [Understanding Impacts of High-Order Loss Approximations and Features in Deep Learning Interpretation](https://arxiv.org/abs/1902.00407)



# packages

- [iNNvestigate neural nets](https://arxiv.org/abs/1808.04260) - provides a common interface and out-of-thebox implementation
- [tensorfuzz](https://arxiv.org/abs/1807.10875) - debugging
- [interactive explanation tool](https://link.springer.com/chapter/10.1007/978-3-030-13463-1_6)

# reviews

- [Interpretable machine learning: definitions, methods, and applications](https://arxiv.org/abs/1901.04592)
- [Interpretable Deep Learning in Drug Discovery](https://arxiv.org/abs/1903.02788)
- [Explanation Methods in Deep Learning: Users, Values, Concerns and Challenges](https://arxiv.org/abs/1803.07517)

## criticisms / eval

- [Sanity Checks for Saliency Maps](https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps.pdf)
- [Quantifying Interpretability of Arbitrary Machine Learning Models Through Functional Decomposition](https://arxiv.org/pdf/1904.03867.pdf)
- [Evaluating Feature Importance Estimates](https://arxiv.org/abs/1806.10758)
- [Interpretable Deep Learning under Fire](https://arxiv.org/abs/1812.00891)

# cnns

- good summary: https://distill.pub/2017/feature-visualization/


- **visualize intermediate features**
    1. visualize filters by layer
      - doesn't really work past layer 1
    2. *decoded filter* - rafegas & vanrell 2016
      - project filter weights into the image space
      - pooling layers make this harder
    3. *deep visualization* - yosinski 15
- penalizing activations
    - tsang interpreatble cnns
    - teaching compositionality to cnns - mask features by objects
- maximal activation stuff
    1. images that maximally activate a feature 
      - *deconv nets* - Zeiler & Fergus (2014)
        - might want to use optimization to generate image that makes optimal feature instead of picking from training set
      - before this, erhan et al. did this for unsupervised features
    2. deep dream - reconstruct image from feature map
      - blog: https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
      - could use natural image prior
      - could train deconvolutional NN
      - also called *deep neuronal tuning* - GD to find image that optimally excites filters
    3. define *neuron feature* - weighted average version of a set of maximum activation images that capture essential properties - rafegas_17
      - can also define *color selectivity index* - angle between first PC of color distribution of NF and intensity axis of opponent color space
      - *class selectivity index* - derived from classes of images that make NF
    4. saliency maps for each image / class
      - simonyan et al 2014
- attention maps
    1. occluding parts of the image
      - sweep over image and remove patches
      - which patch removals had highest impact on change in class?
    2. text usually uses attention maps
      - ex. karpathy et al LSTMs
      - ex. lei et al. - most relevant sentences in sentiment prediction
- concept activation vectors

    - [concept activation vectors](https://arxiv.org/abs/1711.11279)

        - Given: a user-defined set of examples for a concept (e.g., ‘striped’), and random
            examples, labeled training-data examples for the studied class (zebras) 
        - given trained network
        - TCAV can quantify the model’s sensitivity to the concept for that class. CAVs are learned by training a linear classifier to distinguish between the activations produced by
            a concept’s examples and examples in any layer
        - CAV - vector orthogonal to the classification boundary
        - TCAV uses the derivative of the CAV direction wrt input
    - [automated concept activation vectors](https://arxiv.org/abs/1902.03129)

        - Given a set of concept discovery images, each image is segmented with different resolutions
            to find concepts that are captured best at different sizes. (b) After removing duplicate segments, each segment is resized tothe original input size resulting in a pool of resized segments of the discovery images. (c) Resized segments are mapped to a model’s activation space at a bottleneck layer. To discover the concepts associated with the target class, clustering with outlier removal is performed. (d) The output of our method is a set of discovered concepts for each class, sorted by their importance in prediction
- **posthoc methods**
    1. ribeiro's LIME model - local approximation to the model
    2. dosovitskiy et al 16 - train generative deconv net to create images from neuron activations
      - aubry & russel 15 do similar thing
    3. gradient-based methods - visualize what in image would change class label
      - *guided backpropagation* - springenberg et al
        - lets you better create maximally specific image
      - selvaraju 17 - *grad-CAM*
    4. koh and liang 17 - *find training points* that contribute most to classification errors

## textual explanations

1. hendricks et al
2. darrell vision + textual work
3. [Adversarial Inference for Multi-Sentence Video Description](https://arxiv.org/pdf/1812.05634.pdf) - adversarial techniques during inference for a better multi-sentence video description
4. [Object Hallucination in Image Captioning](https://aclweb.org/anthology/D18-1437) - image relevance metric - asses rate of object hallucination
   1. CHAIR metric - what proportion of words generated are actually in the image according to gt sentences and object segmentations
5. [women also snowboard](https://arxiv.org/pdf/1803.09797.pdf) - force caption models to look at peopl when making gender-specific predictions
6. [Fooling Vision and Language Models Despite Localization and Attention Mechanism](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Fooling_Vision_and_CVPR_2018_paper.pdf) -  can do adversarial attacks on captioning and VQA
7. [Grounding of Textual Phrases in Images by Reconstruction](https://arxiv.org/pdf/1511.03745.pdf) - given text and image provide a bounding box (supervised problem w/ attention)

## misc

1. 7. 
2. create an explanatory graph
  - zhang_17 - create a graph that responds better to things like objects than individual neurons
3. information bottleneck framework on deep nets (schwartz-ziv & tishby)
4. *t-SNE* embeds images into a clustered 2d space so we can see them
5. wei_15 understanding intraclass variation
6. mahendran_14 inverting CNNS

## model-based
1. learning AND-OR Templates for Object Recognition and Detection (zhu_13)
2. ross et al. - constraing model during training
3. scat transform idea (mallat_16 rvw, oyallan_17)
4. force interpretable description by piping through something interpretable (ex. tenenbaum scene de-rendering)
5. learn concepts through probabilistic program induction
6. force biphysically plausible learning rules



## interpreting weights

-  tsang_17: interacting inputs must follow strongly weighted connections to a common hidden unit before the final output

## prototypes

- [prototypes I](https://arxiv.org/pdf/1710.04806.pdf)
  - uses encoder/decoder setup
  - encourage every prototype to be similar to at least one encoded input
  - learned prototypes in fact look like digits
  - correct class prototypes go to correct classes
  - loss: classification + reconstruction + distance to a training point
- [prototypes II](https://arxiv.org/abs/1806.10574)
  - can have smaller prototypes
  - l2 distance
  - require the filters to be identical to the latent representation of some training image patch
  - cluster image patches of a particular class around the prototypes of the same class, while separating image patches of different classes
  - maxpool class prototypes so spatial size doesn't matter
  - also get heatmap of where prototype was activated (only max really matters)
  - train in 3 steps
    - train everything: classification + clustering around intraclass prototypes + separation between interclass prototypes (last layer fixed to 1s / -0.5s)
    - project prototypes to data patches
    - learn last layer
- [posthoc prototypes](https://openreview.net/forum?id=r1xyx3R9tQ)
- **counterfactual explanations** - like adversarial, counterfactual explanation describes smallest change to feature vals that changes the prediction to a predefined output
  - maybe change fewest number of variables not their values
  - counterfactual should be reasonable (have likely feature values)
  - human-friendly
  - usually multiple possible counterfactuals (Rashomon effect)
  - can use optimization to generate counterfactual
  - **anchors** - opposite of counterfactuals, once we have these other things won't change the prediction
- prototypes (assumed to be data instances)
  - prototype = data instance that is representative of lots of points
  - criticism = data instances that is not well represented by the set of prototypes
  - examples: k-medoids or MMD-critic
    - selects prototypes that minimize the discrepancy between the data + prototype distributions

# high-level

- hooker_17_ml_and_future
  - anti-realism over realism
  - lack of interpretability in NNs is part of what makes them powerful
  - *naked predictions* - numbers with no real interpretation
    - more central to science than modelling?
    - no theory needed? (Breiman 2001)
- old school: realist studied neuroscience (Wundt), anti-realist just stimuli/response patterns (Skinner), now neither
- interpretability properties
  - *simplicity* - too complex
  - *risk* - too complex
  - *efficiency* - basically generalizability
  - *unification* - answers *ontology* - the nature of being
  - *realism* in a partially accessible world
- overall, they believe there is inherent value of ontological description



# evaluating interp

- [An Evaluation of the Human-Interpretability of Explanation](https://arxiv.org/pdf/1902.00006.pdf)
- [How do Humans Understand Explanations from Machine Learning Systems?: An Evaluation of the Human-Interpretability of Explanation](https://arxiv.org/pdf/1802.00682.pdf)
- [Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/pdf/1702.08608.pdf)

# feature importance and interactions

- build-up = context-free, less faithful: score is contribution of only variable of interest ignoring other variables
- break-down = context-dependent, more faithful: score is contribution of variable of interest given all other variables (e.g. permutation test - randomize var of interest from right distr.)

## interactions explicitly

- H-statistic*: 0 for no interaction, 1 for complete interaction
  - how much of the variance of the output of the joint partial dependence is explained by the interaction instead of the individuals
  - $H^2_{jk} = \underbrace{\sum_i [\overbrace{PD(x_j^{(i)}, x_k^{(i)})}^{\text{interaction}} \overbrace{- PD(x_j^{(i)}) - PD(x_k^{(i)})}^{\text{individual}}]^2}_{\text{sum over data points}} \: / \: \underbrace{\sum_i [PD(x_j^{(i)}, x_k^{(i)})}_{\text{normalization}}]^2$
  - same assumptions as PDP: features need to be independent
- alternatives
  - variable interaction networks (Hooker, 2004) - decompose pred into main effects + feature interactions
  - PDP-based feature interaction (greenwell et al. 2018)

## feature importance

- importance of a feature is the increase in the prediction error after we permuted the feature's values
- If features are correlated, the permutation feature importance can be biased by unrealistic data
  instances (PDP problem)
- not the same as model variance
- Adding a correlated feature can decrease the importance of the associated feature

## surrogates

- could globally fit a simpler model to the complex model
- **local surrogate (LIME)** - fit a simple model locally to on point and interpret that
  - select data perturbations and get new predictions
    - for images, this is turning superpixels on/off
    - superpixels determined in unsupervised way
  - weight the new samples based on their proximity
  - train a weighted, interpretable model on these points

## shapley values

- **shapley value** - average marginal contribution of a feature value across all possible sets of feature values
  - "how much does prediction change on average when this feature is added?"
  - tells us the difference between the actual prediction and the average prediction
  - estimating: all possible sets of feature values have to be evaluated with and without the j-th feature
    - this includes sets of different sizes
- shapley sampling value - sample instead of exactly computing
  - quantitative input influence is similar to this...
- [A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predicti)
  - 3 properties
    - local accuracy - basically, explanation scores sum to original prediction
    - missingness - features with $x'_i=0$ have 0 impact
    - consistency - if a model changes so that some simplified input’s contribution increases or stays the same regardless of the other inputs, that input’s attribution should not decrease.
  - ![Screen Shot 2019-05-28 at 10.23.49 PM](../../Desktop/Screen Shot 2019-05-28 at 10.23.49 PM.png)

## example-based explanations

- influential instances - want to find important data points
  - deletion diagnostics - delete a point and see how much it changed

  - koh 17 influence funcs: use **Hessian** ($\theta x \theta$) to give effect of upweighting a point

  - influence functions = inifinitesimal approach - upweight one person by infinitesimally small weight and see how much estimate changes (e.g. calculate first derivative)

    - influential instance - when data point removed, has a strong effect on the model (not necessarily same as an outlier)
    - requires access to gradient (e.g. nn, logistic regression)
    - take single step with Newton's method after upweighting loss

    - yield change in parameters by removing one point
    - yield change in loss at one point by removing a different point (by multiplying above by cahin rule)
    - yield change in parameters by modifying one point


# model-agnostic

- **pdp plots** - marginals (force value of plotted var to be what you want it to be)
- separate into **ice plots**  - marginals for instance

  - average of ice plots = pdp plot
  - sometimes these are centered, sometimes look at derivative
- both pdp ice suffer from many points possibly not being real
- possible solution: **Marginal plots M-plots** (bad name - uses conditional, not marginal)

  - only use points conditioned on certain variable
  - problem: this bakes things in (e.g. if two features are correlated and only one important, will say both are important)
- **ALE-plots** - take points conditioned on value of interest, then look at differences in predictions around a window

  - this gives pure effect of that var and not the others
  - needs an order (i.e. might not work for caterogical)
  - doesn't give you individual curves
  - recommended very highly by the book...
  - they integrate as you go...
- summary: To summarize how each type of plot (PDP, M, ALE) calculates the effect of a feature at a certain grid value v:
   - Partial Dependence Plots: “Let me show you what the model predicts on average when each data instance has the value v for that feature. I ignore whether the value v makes sense for all data instances.” 
- M-Plots: “Let me show you what the model predicts on average for data instances that have values close to v for that feature. The effect could be due to that feature, but also due to correlated features.” 
  - ALE plots: “Let me show you how the model predictions change in a small “window” of the feature around v for data instances in that window.” 
- [What made you do this? Understanding black-box decisions with sufficient input subsets](https://arxiv.org/pdf/1810.03805.pdf)
   - want to find smallest subsets of features which can produce the prediction
      - other features are masked or imputed



# fairness

- good introductory [blog](https://towardsdatascience.com/a-tutorial-on-fairness-in-machine-learning-3ff8ba1040cb)
- causes of bias
  - skewed sample
  - tainted examples
  - selectively limited features
  - sample size disparity
  - proxies of sensitive attributes
- definitions
  - Unawareness - don't show sensitive attributes
    - flaw: other attributes can still signal for it
  - group fairness
    - Demographic Parity - means for each group should be approximately equal
      - flaw: means might not be equal
    - Equalized Odds - predictions are independent of group given label
      - equality of opportunity: $p(\hat y=1|y=1)$ is same for both groups
    - Predictive Rate Parity - Y is independent of group given prediction
  - Individual Fairness
    - similar individuals should be treated similarly
  - Counterfactual fairness
    - replace attributes w/ flipped values
- fair algorithms
  - preprocessing - remove sensitive information
  - optimization at training time - add regularization
  - postprocessing - change thresholds to impose fairness