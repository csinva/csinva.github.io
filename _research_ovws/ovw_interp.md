---
layout: notes_without_title
section-type: notes
title: interp
category: research
---

**some papers I like involving interpretable machine learning, largely organized based on this [interpretable ml review](https://arxiv.org/abs/1901.04592), and some notes from this [interpretable ml book](https://christophm.github.io/interpretable-ml-book/)**

[TOC]

# packages

- [iNNvestigate neural nets](https://arxiv.org/abs/1808.04260) - provides a common interface and out-of-thebox implementation
- [tensorfuzz](https://arxiv.org/abs/1807.10875) - debugging
- [interactive explanation tool](https://link.springer.com/chapter/10.1007/978-3-030-13463-1_6)

# reviews

- [Interpretable machine learning: definitions, methods, and applications](https://arxiv.org/abs/1901.04592)
- [model-agnostic debugging book in progress](https://pbiecek.github.io/PM_VEE/preface.html)
- [Interpretable Deep Learning in Drug Discovery](https://arxiv.org/abs/1903.02788)
- [Explanation Methods in Deep Learning: Users, Values, Concerns and Challenges](https://arxiv.org/abs/1803.07517)
- [Towards a Generic Framework for Black-box Explanation Methods](https://hal.inria.fr/hal-02131174v2/document) (henin & metayer 2019)
  - sampling - selection of inputs to submit to the system to be explained
  - generation - analysis of links between selected inputs and corresponding outputs to generate explanations
  
    1. *proxy* - approximates model (ex. rule list, linear model)
	  
    2. *explanation generation* - explains the proxy (ex. just give most important 2 features in rule list proxy, ex. LIME gives coefficients of linear model, Shap: sums of elements)
  - interaction (with the user)
  - this is a super useful way to think about explanations (especially local), but doesn't work for SHAP / CD which are more about how much a variable contributes rather than a local approximation

![Screen Shot 2019-06-04 at 11.38.50 AM](assets/Screen Shot 2019-06-04 at 11.38.50 AM.png)

![Screen Shot 2019-06-04 at 11.41.02 AM](assets/Screen Shot 2019-06-04 at 11.41.02 AM.png)

![Screen Shot 2019-06-04 at 11.42.17 AM](assets/Screen Shot 2019-06-04 at 11.42.17 AM.png)

- [feature (variable) importance measurement review (VIM)](https://www.sciencedirect.com/science/article/pii/S0951832015001672) (wei et al. 2015)
  - often-termed sensitivity, contribution, or impact
  - some of these can be applied to data directly w/out model (e.g. correlation coefficient, rank correlation coefficient, moment-independent VIMs)
  - ![Screen Shot 2019-06-14 at 9.07.18 AM](assets/Screen Shot 2019-06-14 at 9.07.18 AM.png)
  - (epistemic) uncerainty: given model and distrs. on the input vars
    - generate samples of input vars
    - compute outputs for these samples
    - characterize uncertainty of output
  - types
    - difference-based - deriv=based methods, local importance measure, morris' screening method
      - **LIM** (local importance measure) - like LIME
        - can normalize weights by values of x, y, or ratios of their standard deviations
        - can also decompose variance to get the covariances between different variables
        - can approximate derivative via adjoint method or smth else
      - **morris' screening method**
        - take a grid of local derivs and look at the mean / std of these derivs
        - can't distinguish between nonlinearity / interaction
      - using the squared derivative allows for a close connection w/ sobol's total effect index
        - can extend this to taking derivs wrt different combinations of variables
    - parametric regression
      - correlation coefficient, linear reg coeffeicients
      - **partial correlation coefficient** (PCC) - wipe out correlations due to other variables
        - do a linear regression using the other variables (on both X and Y) and then look only at the residuals
      - rank regression coefficient - better at capturing nonlinearity
      - could also do polynomial regression
      - more techniques (e.g. relative importance analysis RIA)
    - nonparametric regression
      - use something like LOESS, GAM, projection pursuit
      - rank variables by doing greedy search (add one var at a time) and seeing which explains the most variance
    - hypothesis test
      - **grid-based hypothesis tests**: splitting the sample space (X, Y) into grids and then testing whether the patterns of sample distributions across different grid cells are random
        - ex. see if means vary
        - ex. look at entropy reduction
      - other hypothesis tests include the squared rank difference, 2D kolmogorov-smirnov test, and distance-based tests
    - variance-based vim (sobol's indices)
      - sobol's indices - attribute total variance of model output: $Y = g(\mathbf{X}) = g_0 + \sum_i g_i (X_i) + \sum_i \sum_{j > i} g_{ij} (X_i, X_j) + \dots + g_{1,2,..., n}$
        - $g_0 = \mathbf E (Y), \:g_i = \mathbf E(Y|X_i) - g_0, \:g_{ij} = \mathbf E (Y|X_i, X_j) - g_i - g_j - g_0$
        - take variances of these terms
        - if there are correlations between variables some of these terms can misbehave
      - $S_i$: Sobol’s main effect index for $i$ measures the average residual variance of model output when all the inputs except $X_i$ are fixed over their full supports
        - small value indicates $X_i$ is non-influential
        - usually used to select important variabels
      - $S_{Ti}$: Sobol's total effect index - include all terms (even interactions) involving a variable
        - usually used to screen unimportant variables
    - moment-independent vim
      - want more than just the variance ot the output variables
      - e.g. **delta index** = average dist. between $f_Y(y)$ and $f_{Y|X_i}(y)$ when $X_i$ is fixed over its full distr.
        - $\delta_i = \frac 1 2 \mathbb E \int |f_Y(y) - f_{Y|X_i} (y) | dy = \frac 1 2 \int \int |f_{Y, X_i}(y, x_i) - f_Y(y) f_{X_i}(x_i)|dy \,dx_i$
        - moment-independent because it depends on the density, not just any moment (like measure of dependence between $y$ and $X_i$
    - graphic vim - like curves
      - e.g. scatter plot, meta-model plot, regional VIMs, parametric VIMs
      - CSM - relative change of model ouput mean when range of $X_i$ is reduced to any subregion
      - CSV - same thing for variance
  - a lot of algos for sampling
- vim definition
  1. a quantitative indicator that quantifies the change of model output value w.r.t. the change or permutation of one or a set of input variables
  2. an indicator that quantifies the contribution of the uncertainties of one or a set of input variables to the uncertainty of model output variable
  3. an indicator that quantifies the strength of dependence between the model output variable and one or a set of input variables. 

## criticisms / eval

- [Sanity Checks for Saliency Maps](https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps.pdf)
- [Quantifying Interpretability of Arbitrary Machine Learning Models Through Functional Decomposition](https://arxiv.org/pdf/1904.03867.pdf)
- [Evaluating Feature Importance Estimates](https://arxiv.org/abs/1806.10758)
- [Interpretable Deep Learning under Fire](https://arxiv.org/abs/1812.00891)
- [An Evaluation of the Human-Interpretability of Explanation](https://arxiv.org/pdf/1902.00006.pdf)
- [How do Humans Understand Explanations from Machine Learning Systems?: An Evaluation of the Human-Interpretability of Explanation](https://arxiv.org/pdf/1802.00682.pdf)
- [Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/pdf/1702.08608.pdf)
- [Manipulating and Measuring Model Interpretability](https://arxiv.org/abs/1802.07810)
- [Evaluating Explanation Without Ground Truth in Interpretable Machine Learning](https://arxiv.org/pdf/1907.06831.pdf)
  - 3 criteria
    - predictability (does the knowledge in the explanation generalize well)
    - fidelity (does explanation reflect the target system well)
    - persuasibility (does human satisfy or comprehend explanation well)

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
    - [interpretable cnns](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0490.pdf) (zhang et al. 2018) - penalize activations to make filters slightly more intepretable
      - could also just use specific filters for specific classes...
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
    5. [Diagnostic Visualization for Deep Neural Networks Using Stochastic Gradient Langevin Dynamics](https://arxiv.org/pdf/1812.04604.pdf)
- attention maps
    1. occluding parts of the image
      - sweep over image and remove patches
      - which patch removals had highest impact on change in class?
    2. text usually uses attention maps
      - ex. karpathy et al LSTMs
      - ex. lei et al. - most relevant sentences in sentiment prediction
    3. class-activation map - sum the activations across channels (weighted by their weight for a particular class)
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
    1. dosovitskiy et al 16 - train generative deconv net to create images from neuron activations
       - aubry & russel 15 do similar thing
    3. gradient-based methods - visualize what in image would change class label
      - gradient * input
      - integrated gradients
      - lrp
      - taylor decomposition
      - deeplift
      - guided backpropagation - springenberg et al
        - lets you better create maximally specific image
      - selvaraju 17 - *grad-CAM*
      - [grad-cam++](https://arxiv.org/abs/1710.11063)
      - [competitive gradients](https://arxiv.org/pdf/1905.12152.pdf) (gupta & arora 2019)
        - Label  "wins" a pixel if either (a) its map assigns that pixel a positive score higher than the scores assigned by every other label ora negative score lower than the scores assigned by every other label. 
        - final saliency map consists of scores assigned by the chosen label to each pixel it won, with the map containing a score 0 for any pixel it did not win.
        - can be applied to any method which satisfies completeness (sum of pixel scores is exactly the logit value)
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

1. create an explanatory graph
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



# tree ensembles

- MDI = Gini import
- Breiman proposes permutation tests: Breiman, Leo. 2001. “Random Forests.” Machine Learning 45 (1). Springer: 5–32
- [Explainable AI for Trees: From Local Explanations to Global Understanding](https://arxiv.org/abs/1905.04610)
  - shap-interaction scores - distribute among pairwise interactions + local effects
  - plot lots of local interactions together - helps detect trends
  - propose doing shap directly on loss function (identify how features contribute to loss instead of prediction)
  - can run supervised clustering (where SHAP score is the label) to get meaningful clusters
    - alternatively, could do smth like CCA on the model output
- [conditional variable importance for random forests](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-307)
  - propose permuting conditioned on the values of variables not being permuted
    - to find region in which to permute, define the grid within which the values of $X_j$ are permuted for each tree by means of the partition of the feature space induced by that tree
  - many scores (such as MDI, MDA) measure marginal importance, not conditional importance
    - as a result, correlated variables get importances which are too high
- [treeshap](https://arxiv.org/abs/1802.03888): prediction-level
  - individual feature attribution: want to decompose prediction into sum of attributions for each feature
    - each thing can depend on all features
  - Saabas method: basic thing for tree
    - you get a pred at end
    - count up change in value at each split for each variable
  - three properties
    - local acc - decomposition is exact
    - missingness - features that are already missing are attributed no importance
      - for missing feature, just (weighted) average nodes from each split
    - consistency - if F(X) relies more on a certain feature j, $F_j(x)$ should 
      - however Sabaas method doesn't change $F_j(X)$ for $F'(x) = F(x) + x_j$
  - these 3 things iply we want shap values
  - average increase in func value when selecting i (given all subsets of other features)
  - for binary features with totally random splits, same as Saabas
  - **can cluster based on explanation similarity** (fig 4)
    - can quantitatively evaluate based on clustering of explanations
  - their fig 8 - qualitatively can see how different features alter outpu
  - gini importance is like weighting all of the orderings
- [understanding variable importances in forests of randomized trees](http://papers.nips.cc/paper/4928-understanding-variable-importances-in-forests-of-randomized-tre) (louppe et al. 2013)
  - consider fully randomized trees
    - assume all categorical
    - randomly pick feature at each depth, split on all possibilities
    - also studied by biau 2012
    - extreme case of random forest w/ binary vars?
  - real trees are harder: correlated vars and stuff mask results of other vars lower down
  - asymptotically, randomized trees might actually be better

# rule lists / sets

- these algorithms usually don't support regression, but you can get regression by cutting the outcome into intervals
- oneR algorithm - select feature that carries most information about the outcome and then split multiple times on that feature
- sequential covering - keep trying to cover more points sequentially
- people are frequently trying to extract rules from trained dnns
- [foundations of rule learning](https://dl.acm.org/citation.cfm?id=2788240) (furnkranz et al. 2014)
  - 2 basic concepts for a rule
    - converage = support
    - accuracy = confidence = consistency
      - measures for rules: precision, info gain, correlation, m-estimate, Laplace estimate
- [interpretable classifiers using rules and bayesian analysis](https://projecteuclid.org/download/pdfview_1/euclid.aoas/1446488742) (letham et al. 2015)
  - start by mining rules (want them to apply to a large amount of data and not have too many conditions) - uses FP-Growth algorithm (borgelt 2005), could also uses Apriori or Eclat
    - current approach does not allow for negation (e.g. not diabetes) and must split continuous variables into categorical somehow (e.g. quartiles)
    - mines things that frequently occur together, but doesn't look at outcomes in this step - okay (since this is all about finding rules with high support)
  - learn rules w/ prior for short rule conditions and short lists
    - start w/ random list 
    - sample new lists by adding/removing/moving a rule
    - at the end, return the list that had the highest probability
  - [scalable bayesian rule lists](https://dl.acm.org/citation.cfm?id=3306086) (yang et al. 2017) - faster algorithm for computing
- [learning certifiably optimal rules lists](https://dl.acm.org/citation.cfm?id=3098047) (angelino et al. 2017) - optimization for categorical feature space
  - can get upper / lower bounds for loss = risk + $\lambda$ * listLength
- [interpretable decision set](https://dl.acm.org/citation.cfm?id=2939874) (lakkaraju et al. 2016) - set of if then rules which are all independent (not falling)
- [A Bayesian Framework for Learning Rule Sets for Interpretable Classification](http://www.jmlr.org/papers/volume18/16-003/16-003.pdf) (wang et al. 2017) - rules are a bunch of clauses OR'd together (e.g. if (X1>0 AND X2<1) OR (X2<1 AND X3>1) OR ... then Y=1)
- [optimal sparse decision trees](https://arxiv.org/abs/1904.12847) (hu et al. 2019) - optimal decision trees for binary variables
- [2helps2b paper](https://www.ncbi.nlm.nih.gov/pubmed/29052706)
  - ![Screen Shot 2019-06-11 at 11.17.35 AM](assets/Screen Shot 2019-06-11 at 11.17.35 AM.png)

# posthoc model-agnostic methods

1. local surrogate ([LIME](https://arxiv.org/abs/1602.04938)) - fit a simple model locally to on point and interpret that
   - select data perturbations and get new predictions
     - for tabular data, this is just varying the values around the prediction
     - for images, this is turning superpixels on/off
     - superpixels determined in unsupervised way
   - weight the new samples based on their proximity
   - train a kernel-weighted, interpretable model on these points
   - LEMNA - like lime but uses lasso + small changes
2. [anchors](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16982/15850) (ribeiro et al. 2018) - find biggest square region of input space that contains input and preserves same output (with high precision)
   1. does this search via iterative rules
3. [What made you do this? Understanding black-box decisions with sufficient input subsets](https://arxiv.org/pdf/1810.03805.pdf)
   - want to find smallest subsets of features which can produce the prediction
     - other features are masked or imputed
4. [VIN](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.91.7500&rep=rep1&type=pdf) (hooker 04) - variable interaction networks - globel explanation based on detecting additive structure in a black-box, based on ANOVA
5. [local-gradient](http://www.jmlr.org/papers/v11/baehrens10a.html) (bahrens et al. 2010) - direction of highest slope towards a particular class / other class
6. [golden eye](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/article/10.1007/s10618-014-0368-8&casa_token=AhKnW6Xx4L0AAAAA:-SEMsMjDX3_rU5gyGx6plcmF5A_ufXvsWJHzjCUIGWHGW0fqOe50yhWKYOK6UIPDHQaUwEkE3RK17XOByzo) (henelius et al. 2014) - randomize different groups of features and search for groups which interact
7. model distillation
   1. Trepan - approximate model w/ a decision tree
   2. [BETA](https://arxiv.org/abs/1707.01154) (lakkaraju et al. 2017) - approximate model by a rule list
8. **[shapley value](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predicti)** - average marginal contribution of a feature value across all possible sets of feature values
  - "how much does prediction change on average when this feature is added?"
  - tells us the difference between the actual prediction and the average prediction
  - estimating: all possible sets of feature values have to be evaluated with and without the j-th feature
    - this includes sets of different sizes
    - to evaluate, take expectation over all the other variables, fixing this variables value
  - shapley sampling value - sample instead of exactly computing
    - quantitative input influence is similar to this...
  - satisfies 3 properties
      - local accuracy - basically, explanation scores sum to original prediction
      - missingness - features with $x'_i=0$ have 0 impact
      - consistency - if a model changes so that some simplified input’s contribution increases or stays the same regardless of the other inputs, that input’s attribution should not decrease.
  - interpretation: Given the current set of feature values, the contribution of a feature value to the
    difference between the actual prediction and the mean prediction is the estimated Shapley value.
  - recalculate via sampling other features in expectation
9. [quantitative input influence](https://ieeexplore.ieee.org/abstract/document/7546525) - similar to shap but more general
10. permutation importance - increase in the prediction error after we permuted the feature's values
  - If features are correlated, the permutation feature importance can be biased by unrealistic data
  instances (PDP problem)
  - not the same as model variance
  - Adding a correlated feature can decrease the importance of the associated feature
11. [L2X: information-theoretical local approximation](https://arxiv.org/pdf/1802.07814.pdf) (chen et al. 2018) - locally assign feature importance based on mutual information with function
12. [Learning Explainable Models Using Attribution Priors + Expected Gradients](https://arxiv.org/abs/1906.10670) - like doing integrated gradients in many directions (e.g. by using other points in the training batch as the baseline)
    - can use this prior to help improve performance

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

# model-agnostic curves

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

# trust scores / uncertainty

*papers using embeddings to generate confidences*

- [been kim trust paper](http://papers.nips.cc/paper/7798-to-trust-or-not-to-trust-a-classifier.pdf) - trust score uses density over some set of nearest neighbors(do clustering for each class - trust score = distance to once class's cluster vs the other classes')
  - [papernot knn](https://arxiv.org/abs/1803.04765)
  - [distance-based confidence scores](https://arxiv.org/pdf/1709.09844.pdf)
  - [deep kernel knn](https://arxiv.org/pdf/1811.02579.pdf)
  - fair paper: gradients should be larger if you are on the image manifold
- [outlier-detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
    - [isolation forest](https://ieeexplore.ieee.org/abstract/document/4781136) - lower average number of random splits required to isolate a sample means more outlier
- lots of papers on confidence calibration (transforms outputs into probabilities)
- [get confidences before overfitting](https://arxiv.org/abs/1805.08206)
    - 2 popular things: max margin, entropy of last layer
    - [add an extra output for uncertainty](https://arxiv.org/abs/1810.01861)
    - [learn to predict confidences](https://arxiv.org/pdf/1802.04865.pdf)
- also methods on predict with rejection possibility

  - [contextual outlier detection](https://arxiv.org/abs/1711.10589)
  - [ensembling background](https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/)
- [How to Generate Prediction Intervals with Scikit-Learn and Python](https://towardsdatascience.com/how-to-generate-prediction-intervals-with-scikit-learn-and-python-ab3899f992ed)
  - can use quantile loss to penalize models differently
  - than can use these different models to get confidence intervals
  - [can easily do this with sklearn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)
  - quantile loss = $\begin{cases} \alpha \cdot \Delta & \text {if} \quad \Delta > 0\\(\alpha - 1) \cdot \Delta & \text{if} \quad \Delta < 0\end{cases}$
    - $\Delta =$ actual - predicted
    - ![Screen Shot 2019-06-26 at 10.06.11 AM](assets/Screen Shot 2019-06-26 at 10.06.11 AM.png)
- [uncertainty though ensemble confidence](http://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles)
  - predict mean and variance w/ each network then ensemble
  - also add in adversarial training
  - [snapshot ensembles](https://arxiv.org/abs/1704.00109)
- bayesian neural nets
  - [icu bayesian dnns](https://aiforsocialgood.github.io/icml2019/accepted/track1/pdfs/38_aisg_icml2019.pdf)
    - focuses on epistemic uncertainty
    - could use one model to get uncertainty and other model to predict
  - [Evaluating Scalable Bayesian Deep Learning Methods for Robust Computer Vision](https://arxiv.org/pdf/1906.01620.pdf)
    - **epistemic uncertainty** - uncertainty in the DNN model parameters
      - without good estimates of this, often get aleatoric uncertainty wrong (since $p(y|x) = \int p(y|x, \theta) p(\theta |data) d\theta$
    - **aleatoric uncertainty** -  inherent and irreducible data noise (e.g. features contradict each other)
      - this can usually be gotten by predicting a distr. $p(y|x)$ instead of a point estimate
      - ex. logistic reg. already does this
      - ex. regression - just predict mean and variance of Gaussian
- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](http://proceedings.mlr.press/v48/gal16.pdf)
  - dropout at test time gives you uncertainty



# fairness

- good introductory [blog](https://towardsdatascience.com/a-tutorial-on-fairness-in-machine-learning-3ff8ba1040cb)
- causes of bias
  - skewed sample
  - tainted examples
  - selectively limited features
  - sample size disparity
  - proxies of sensitive attributes
- definitions
  - unawareness - don't show sensitive attributes
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

# data science

- [experiment-tracking frameworks](https://www.reddit.com/r/MachineLearning/comments/bx0apm/d_how_do_you_manage_your_machine_learning/)

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
  
- [Interpreting Neural Network Judgments via Minimal, Stable, and Symbolic Corrections](https://arxiv.org/pdf/1802.07384.pdf)

- [DeepPINK: reproducible feature selection in deep neural networks](https://arxiv.org/pdf/1809.01185.pdf)

- "Transparency by Disentangling Interactions"

- "To Trust Or Not To Trust A Classifier"

- [Towards Robust Interpretability with Self-Explaining Neural Networks](https://arxiv.org/pdf/1806.07538.pdf)

- "Explaining Deep Learning Models -- A Bayesian Non-parametric Approach"

- [Detecting Potential Local Adversarial Examples for Human-Interpretable Defense](https://arxiv.org/pdf/1809.02397.pdf)

- [Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning](https://arxiv.org/pdf/1803.04765.pdf)
  
  - [Interpreting Neural Networks With Nearest Neighbors](https://arxiv.org/pdf/1809.02847.pdf)
  
- [Interpreting Layered Neural Networks via Hierarchical Modular Representation](https://arxiv.org/pdf/1810.01588.pdf)

- [Entropic Variable Boosting for Explainability & Interpretability in Machine Learning](https://arxiv.org/abs/1810.07924)

- [Explain to Fix: A Framework to Interpret and Correct DNN Object Detector Predictions](https://arxiv.org/pdf/1811.08011.pdf)

- [Understanding Individual Decisions of CNNs via Contrastive Backpropagation](https://arxiv.org/abs/1812.02100v1)

- [“What are You Listening to?” Explaining Predictions of Deep Machine Listening Systems](https://ieeexplore.ieee.org/abstract/document/8553178)

- importance scores
  - [Variable Importance Clouds: A Way to Explore Variable Importance for the Set of Good Models](https://arxiv.org/pdf/1901.03209.pdf) 
  - [Permutation tests in general](http://arxiv.org/abs/1801.01489): Fisher, Aaron, Cynthia Rudin, and Francesca Dominici. 2018. “Model Class Reliance: Variable Importance Measures for any Machine Learning Model Class, from the ‘Rashomon’ Perspective.”
  
- [Recovering Pairwise Interactions Using Neural Networks](https://arxiv.org/pdf/1901.08361.pdf)

- [How Sensitive are Sensitivity-Based Explanations?](https://arxiv.org/abs/1901.09392)

- [Understanding Impacts of High-Order Loss Approximations and Features in Deep Learning Interpretation](https://arxiv.org/abs/1902.00407)