---
layout: notes
title: transfer learning
category: research
---


{:toc}

See also notes on causal inference for some close connections. 

# overviews

![transfer_taxonomy](../assets/transfer_taxonomy.png) (from [this paper](https://arxiv.org/pdf/2109.14501v1.pdf))

- for neural networks the basic options for transfer-learning are:
  - finetuning the entire model
  - learn a linear layer from features extracted from a single layer (i.e. linear probing)
    - this includes just finetuning the final layer
  - finetune on all the layers
    - [Head2Toe: Utilizing Intermediate Representations for Better Transfer Learning](https://arxiv.org/abs/2201.03529) (evci, et al. 2022) - learn linear layer (using group-lasso) on features extracted from all layers
    - Adapters provide a parameter-efficient alternative to full finetuning in which we can only finetune lightweight neural network layers on top of pretrained weights. [Parameter-Efficient Transfer Learning for NLP](http://proceedings.mlr.press/v97/houlsby19a.html), [AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/abs/2007.07779)

# domain adaptation algorithms

*Domain test bed available [here](https://github.com/facebookresearch/DomainBed), for generalizating to new domains (i.e. performing well on domains that differ from previous seen data)*

- Empirical Risk Minimization (ERM, [Vapnik, 1998](https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034)) - standard training
- Invariant Risk Minimization (IRM, [Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893)) - learns a feature representation such that the optimal linear classifier on top of that representation matches across domains.
- distributional robust optimization
  - instead of minimizing training err, minimize maximum training err over different perturbations
  - Group Distributionally Robust Optimization (GroupDRO, [Sagawa et al., 2020](https://arxiv.org/abs/1911.08731)) - ERM + increase importance of domains with larger errors (see also papers from Sugiyama group e.g. [1](http://papers.neurips.cc/paper/3019-mixture-regression-for-covariate-shift.pdf), [2](https://arxiv.org/abs/1611.02041))
    - minimize error for worst group
  - Variance Risk Extrapolation (VREx, [Krueger et al., 2020](https://arxiv.org/abs/2003.00688)) - encourages robustness over affine combinations of training risks, by encouraging strict equality between training risks
- Interdomain Mixup (Mixup, [Yan et al., 2020](https://arxiv.org/abs/2001.00677)) - ERM on linear interpolations of examples from random pairs of domains + their labels
- Marginal Transfer Learning (MTL, [Blanchard et al., 2011-2020](https://arxiv.org/abs/1711.07910)) - augment original feature space with feature vector marginal distributions and then treat as a supervised learning problem
- Meta Learning Domain Generalization (MLDG, [Li et al., 2017](https://arxiv.org/abs/1710.03463)) - use MAML to meta-learn how to generalize across domains
- learning more diverse predictors
  - Representation Self-Challenging (RSC, [Huang et al., 2020](https://arxiv.org/abs/2007.02454)) - adds dropout-like regularization to important features, forcing model to depend on many features
  - Spectral Decoupling (SD, [Pezeshki et al., 2020](https://arxiv.org/abs/2011.09468)) - regularization which forces model to learn more predictive features, even when only a few suffice
- embedding prior knowledge
  - Style Agnostic Networks (SagNet, [Nam et al., 2020](https://arxiv.org/abs/1910.11645)) - penalize style features (assumed to be spurious)
  - Penalizing explanations ([Rieger et al. 2020](https://arxiv.org/abs/1909.13584)) - penalize spurious features using prior knowledge
- Domain adaptation under structural causal models ([chen & buhlmann, 2020]((https://arxiv.org/abs/2010.15764)))
  - make clearer assumptions for domain adaptation to work
  - introduce CIRM, which works better when both covariates and labels are perturbed in target data
- kernel approach ([blanchard, lee & scott, 2011](https://papers.nips.cc/paper/2011/file/b571ecea16a9824023ee1af16897a582-Paper.pdf)) - find an appropriate RKHS and optimize a regularized empirical risk over the space
- In-N-Out ([xie...lang, 2020](https://arxiv.org/abs/2012.04550)) - if we have many features, rather than using them all as features, can use some as features and some as targets when we shift, to learn the domain shift



## domain invariance

*key idea: want repr. to be invariant to domain label*

- ![Screen Shot 2020-11-10 at 12.05.12 PM](../assets/domain_adv_training.png)
- same idea is used to [learn fair representations](https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf), but domain label is replaced with sensitive attribute
- Domain Adversarial Neural Network (DANN, [Ganin et al., 2015](https://arxiv.org/abs/1505.07818))
- Conditional Domain Adversarial Neural Network (CDANN, [Li et al., 2018](https://arxiv.org/abs/1807.08479)) - variant of DANN matching the conditional distributions  across domains, for all labels 
- Deep CORAL (CORAL, [Sun and Saenko, 2016](https://arxiv.org/abs/1607.01719)) - match mean / covariance of feature distrs
- Maximum Mean Discrepancy (MMD, [Li et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf))
- adversarial discriminative domain adaptation (ADDA [tzeng et al. 2017](https://arxiv.org/abs/1702.05464))
- balancing with [importance weighting](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.370.4921&rep=rep1&type=pdf)
- [Learning Robust Representations by Projecting Superficial Statistics Out](https://arxiv.org/abs/1903.06256) (wang et al. 2019)

## dynamic selection

*Dynamic Selection (DS) refers to techniques in which, for a new test point, pre-trained classifiers are selected/combined from a pool at test time  [review paper](https://www.etsmtl.ca/Unites-de-recherche/LIVIA/Recherche-et-innovation/Publications/Publications-2017/RCruz_InfoFusion.pdf) (cruz et al. 2018), [python package](https://github.com/scikit-learn-contrib/DESlib)*

1. define region of competence
   1. clustering
   2. kNN - more refined than clustering
   3. decision space - e.g. a model's classification boundary, internal splits in a model
   4. potential function - weight all the points (e.g. by their distance to the query point)
2. criteria for selection
   1. individual scores: acc, prob. behavior, rank, meta-learning, complexity
   2. group: data handling, ambiguity, diversity
3. combination
   1. non-trainable: mean, majority vote, product, median, etc.
   2. trainable: learn the combination of models
      1. related: in mixture of experts models + combination are trained jointly
   3. dynamic weighting: combine using local competence of base classifiers
   4. Oracle baseline - selects classifier predicts correct label, if such a classifier exists 

## test-time adaptation

- test-time adaptation
  - test-time augmentation
  - [batch normalization](https://arxiv.org/abs/1603.04779) (AdaBN)
  - [label shift estimation](https://arxiv.org/abs/1802.03916) (BBSE) - $p(y)$ shifts but $P(x|y)$ does not
  - [rotation prediction](https://arxiv.org/abs/1909.13231) (sun et al. 2020)
  - [entropy minimization](https://arxiv.org/abs/2006.10726) (test-time entropy minimization, TENT, wang et al. 2020) - optimize for model confidence (entropy of predictions), using only norm. statistics and channel-wise affine transformations

- combining train-time and test-time adaptation
  - Adaptive Risk Minimization (ARM, [Zhang et al., 2020](https://arxiv.org/abs/2007.02931)) - combines groups at training time + *batches at test-time*
    - *meta-train* the model using simulated distribution shifts, which is enabled by the training groups, such that it exhibits strong *post-adaptation* performance on each shift

# adv attacks

- [Adversarial Attacks and Defenses in Images, Graphs and Text: A Review](https://arxiv.org/abs/1909.08072) (xu et al. 2019) 
- attacks
  - fast gradient step method - keep adding gradient to maximize noise (limit amplitude of pixel's channel to stay imperceptible)
  - [Barrage of Random Transforms for Adversarially Robust Defense](http://openaccess.thecvf.com/content_CVPR_2019/papers/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.pdf) (raff et al. 2019) 
  - [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/abs/1511.04599) (Moosavi-Dezfooli et. al 2016)
- defenses
  - Adversarial training -  training data is augmented with adv examples (Szegedy et al., 2014b; Madry et al., 2017; Tramer et al., 2017; Yu et al., 2019)
    - $$\min _{\boldsymbol{\theta}} \frac{1}{N} \sum_{n=1}^{N} \operatorname{Loss}\left(f_{\theta}\left(x_{n}\right), y_{n}\right)+\lambda\left[\max _{\|\delta\|_{\infty} \leq \epsilon} \operatorname{Loss}\left(f_{\theta}\left(x_{n}+\delta\right), y_{n}\right)\right]$$
    - this perspective differs from "robust statistics" which is usually robustness against some kind of model misspecification/assumptions, not to distr. shift
      - robust stat usually assumes a generative distr. as well
      - still often ends up with the same soln (e.g. ridge regr. corresponds to certain robusteness)
  - Stochasticity: certain inputs or hidden activations are shuffled or randomized (Xie et al., 2017; Prakash et al., 2018; Dhillon et al., 2018)
  - Preprocessing: inputs or hidden activations are quantized, projected into a different representation or are otherwise preprocessed (Guo et al., 2017; Buckman et al., 2018; Kabilan et al., 2018)
  - Manifold projections: an input sample is projected in a lower dimensional space in which the neural network has been trained to be particularly robust (Ilyas et al., 2017; Lamb et al., 2018)
  - Regularization in the loss function: an additional penalty term is added to the optimized objective function to upper bound or to approximate the adversarial loss (Hein and Andriushchenko, 2017; Yan et al., 2018)
  - constraint
    - robustness as a constraint not a loss ([Constrained Learning with Non-Convex Losses](https://arxiv.org/abs/2103.05134) (chamon et al. 2021))
      - $$\begin{aligned}
        \min _{\boldsymbol{\theta}} & \frac{1}{N} \sum_{n=1}^{N} \operatorname{Loss}\left(f_{\theta}\left(x_{n}\right), y_{n}\right) \\
        \text { subject to } & \frac{1}{N} \sum_{n=1}^{N}\left[\max _{\|\delta\|_{\infty} \leq \epsilon} \operatorname{Loss}\left(f_{\theta}\left(\boldsymbol{x}_{n}+\delta\right), y_{n}\right)\right] \leq c
        \end{aligned}$$
      - when penalty is convex, these 2 problems are the same
  - a possible defense against adversarial attacks is to solve the anticausal classification problem by modeling the causal generative direction, a method which in vision is referred to as *analysis by synthesis* ([Schott et al., 2019](https://arxiv.org/abs/1805.09190))
- robustness vs accuracy
  - [robustness may be at odds with accuracy](https://openreview.net/pdf?id=SyxAb30cY7) (tsipiras...madry, 2019)
  - [Precise Tradeoffs in Adversarial Training for Linear Regression](https://arxiv.org/abs/2002.10477) (javanmard et al. 2020) - linear regression with gaussian features
    - use adv. training formula above
  - [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/abs/1901.08573) (Zhang, ..., el ghaoui, Jordan, 2019)
- adversarial examples
  - [Decision Boundary Analysis of Adversarial Examples ](https://pdfs.semanticscholar.org/08c5/88465b7d801ad912ef3e9107fa511ea0e403.pdf)(He, Li, & Song 2019)
  - [Natural Adversarial Examples](https://arxiv.org/abs/1907.07174) (Hendrycks, Zhao, Basart, Steinhardt, & Song 2020)
  - [Image-Net-Trained CNNs Are Biased Towards Texture](https://openreview.net/pdf?id=Bygh9j09KX) (Geirhos et al. 2019)
- adversarial transferability
  - [Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples](https://arxiv.org/abs/1605.07277) (papernot, mcdaniel, & goodfellow, 2016)
  - [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/pdf/1705.07204.pdf) (tramer et al. 2018)
  - [Improving Adversarial Robustness via Promoting Ensemble Diversity](https://arxiv.org/pdf/1901.08846.pdf) (pang et al. 2019)
    - encourage diversity in non-maximal predictions
- robustness
  - smoothness yields robustness (but can be robust without smoothness)
  - margin idea - data points close to the boundary are not robust
    - we want our boundary to go through regions where data is scarce

## nlp

- [QData/TextAttack: TextAttack 🐙 is a Python framework for adversarial attacks, data augmentation, and model training in NLP https://textattack.readthedocs.io/en/master/](https://github.com/QData/TextAttack) (only the attacks on classification tasks, like sentiment classification and entailment)

<table  style="width:100%" border="1">
<thead>
<tr class="header">
<th><strong>Attack Recipe Name</strong></th>
<th><strong>Goal Function</strong></th>
<th><strong>ConstraintsEnforced</strong></th>
<th><strong>Transformation</strong></th>
<th><strong>Search Method</strong></th>
<th><strong>Main Idea</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><code>a2t</code> 
<span class="citation" data-cites="yoo2021a2t"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Word embedding distance, DistilBERT sentence encoding cosine similarity, part-of-speech consistency</sub></td>
<td><sub>Counter-fitted word embedding swap (or) BERT Masked Token Prediction</sub></td>
<td><sub>Greedy-WIR (gradient)</sub></td>
<td ><sub>from "Towards Improving Adversarial Training of NLP Models" (<a href="https://arxiv.org/abs/2109.00544">Yoo et al., 2021)</a></sub></td>
</tr>
<tr>
<td><code>alzantot</code>  <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Language Model perplexity, Word embedding distance</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Genetic Algorithm</sub></td>
<td ><sub>from "Generating Natural Language Adversarial Examples" (<a href="https://arxiv.org/abs/1804.07998">Alzantot et al., 2018</a>)</sub></td>
</tr>
<tr>
<td><code>bae</code> <span class="citation" data-cites="garg2020bae"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>USE sentence encoding cosine similarity</sub></td>
<td><sub>BERT Masked Token Prediction</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>BERT masked language model transformation attack from ("BAE: BERT-based Adversarial Examples for Text Classification" (<a href="https://arxiv.org/abs/2004.01970">Garg & Ramakrishnan, 2019</a>)). </td>
</tr>
<tr>
<td><code>bert-attack</code> <span class="citation" data-cites="li2020bertattack"></span></td>
<td><sub>Untargeted Classification</td>
<td><sub>USE sentence encoding cosine similarity, Maximum number of words perturbed</td>
<td><sub>BERT Masked Token Prediction (with subword expansion)</td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub> ("BERT-ATTACK: Adversarial Attack Against BERT Using BERT" (<a href="https://arxiv.org/abs/2004.09984">Li et al., 2020</a>))</sub></td>
</tr>
<tr>
<td><code>checklist</code> <span class="citation" data-cites="Gao2018BlackBoxGO"></span></td>
<td><sub>{Untargeted, Targeted} Classification</sub></td>
<td><sub>checklist distance</sub></td>
<td><sub>contract, extend, and substitutes name entities</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Invariance testing implemented in CheckList . ("Beyond Accuracy: Behavioral Testing of NLP models with CheckList" (<a href="https://arxiv.org/abs/2005.04118">Ribeiro et al., 2020</a>))</sub></td>
</tr>
<tr>
<td> <code>clare</code> <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>USE sentence encoding cosine similarity</sub></td>
<td><sub>RoBERTa Masked Prediction for token swap, insert and merge</sub></td>
<td><sub>Greedy</sub></td>
<td ><sub>["Contextualized Perturbation for Textual Adversarial Attack" (Li et al., 2020)](https://arxiv.org/abs/2009.07502))</sub></td>
</tr>
<tr>
<td><code>deepwordbug</code> <span class="citation" data-cites="Gao2018BlackBoxGO"></span></td>
<td><sub>{Untargeted, Targeted} Classification</sub></td>
<td><sub>Levenshtein edit distance</sub></td>
<td><sub>{Character Insertion, Character Deletion, Neighboring Character Swap, Character Substitution}</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy replace-1 scoring and multi-transformation character-swap attack (["Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers" (Gao et al., 2018)](https://arxiv.org/abs/1801.04354)</sub></td>
</tr>
<tr>
<td> <code>fast-alzantot</code> <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Language Model perplexity, Word embedding distance</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Genetic Algorithm</sub></td>
<td ><sub>Modified, faster version of the Alzantot et al. genetic algorithm, from (["Certified Robustness to Adversarial Word Substitutions" (Jia et al., 2019)](https://arxiv.org/abs/1909.00986))</sub></td>
</tr>
<tr>
<td><code>hotflip</code> (word swap) <span class="citation" data-cites="Ebrahimi2017HotFlipWA"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>Word Embedding Cosine Similarity, Part-of-speech match, Number of words perturbed</sub></td>
<td><sub>Gradient-Based Word Swap</sub></td>
<td><sub>Beam search</sub></td>
<td ><sub> (["HotFlip: White-Box Adversarial Examples for Text Classification" (Ebrahimi et al., 2017)](https://arxiv.org/abs/1712.06751))</sub></td>
</tr>
<tr>
<td><code>iga</code> <span class="citation" data-cites="iga-wang2019natural"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Word embedding distance</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Genetic Algorithm</sub></td>
<td ><sub>Improved genetic algorithm -based word substitution from (["Natural Language Adversarial Attacks and Defenses in Word Level (Wang et al., 2019)"](https://arxiv.org/abs/1909.06723)</sub></td>
</tr>
<tr>
<td><code>input-reduction</code> <span class="citation" data-cites="feng2018pathologies"></span></td>
<td><sub>Input Reduction</sub></td>
<td></td>
<td><sub>Word deletion</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with word importance ranking , Reducing the input while maintaining the prediction through word importance ranking (["Pathologies of Neural Models Make Interpretation Difficult" (Feng et al., 2018)](https://arxiv.org/pdf/1804.07781.pdf))</sub></td>
</tr>
<tr>
<td><code>kuleshov</code> <span class="citation" data-cites="Kuleshov2018AdversarialEF"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>Thought vector encoding cosine similarity, Language model similarity probability</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Greedy word swap</sub></td>
<td ><sub>(["Adversarial Examples for Natural Language Classification Problems" (Kuleshov et al., 2018)](https://openreview.net/pdf?id=r1QZ3zbAZ)) </sub></td>
</tr>
<tr>
<td><code>pruthi</code> <span class="citation" data-cites="pruthi2019combating"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>Minimum word length, Maximum number of words perturbed</sub></td>
<td><sub>{Neighboring Character Swap, Character Deletion, Character Insertion, Keyboard-Based Character Swap}</sub></td>
<td><sub>Greedy search</sub></td>
<td ><sub>simulates common typos (["Combating Adversarial Misspellings with Robust Word Recognition" (Pruthi et al., 2019)](https://arxiv.org/abs/1905.11268) </sub></td>
</tr>
<tr>
<td><code>pso</code> <span class="citation" data-cites="pso-zang-etal-2020-word"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td></td>
<td><sub>HowNet Word Swap</sub></td>
<td><sub>Particle Swarm Optimization</sub></td>
<td ><sub>(["Word-level Textual Adversarial Attacking as Combinatorial Optimization" (Zang et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.540/)) </sub></td>
</tr>
<tr>
<td><code>pwws</code> <span class="citation" data-cites="pwws-ren-etal-2019-generating"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td></td>
<td><sub>WordNet-based synonym swap</sub></td>
<td><sub>Greedy-WIR (saliency)</sub></td>
<td ><sub>Greedy attack with word importance ranking based on word saliency and synonym swap scores (["Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency" (Ren et al., 2019)](https://www.aclweb.org/anthology/P19-1103/))</sub> </td>
</tr>
<tr>
<td><code>textbugger</code> : (black-box) <span class="citation" data-cites="Li2019TextBuggerGA"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>USE sentence encoding cosine similarity</sub></td>
<td><sub>{Character Insertion, Character Deletion, Neighboring Character Swap, Character Substitution}</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>([(["TextBugger: Generating Adversarial Text Against Real-world Applications" (Li et al., 2018)](https://arxiv.org/abs/1812.05271)).</sub></td>
</tr>
<tr>
<td><code>textfooler</code> <span class="citation" data-cites="Jin2019TextFooler"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Word Embedding Distance, Part-of-speech match, USE sentence encoding cosine similarity</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with word importance ranking  (["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932))</sub> </td>
</tr>
</table>
