---
layout: notes
title: Learning Theory
cat: ml
---

#  learning theory


*references: (1) Machine Learning - Tom Mitchell, (2) An Introduction to Computational Learning Theory - Kearns & Vazirani*

## evolution

- performance is correlation  $Perf_D (h,c) = \sum h(x) \cdot c(x) \cdot P(x)$
  - want $P(Perf_D(h,c) < Perf_D(c,c)-\epsilon) < \delta$

## sample problems

- ex: N marbles in a bag. How many draws with replacement needed before we draw all N marbles?
  - write $P_i = \frac{N-(i-1)}{N}$ where i is number of distinct drawn marbles
    - transition from i to i+1 is geometrically distributed with probability $P_i$
    - mean times is sum of mean of each geometric
  - in order to get probabilities of seeing all the marbles instead of just mean[## draws], want to use Markov's inequailty
- box full of 1e6 marbles
  - if we have 10 evenly distributed classes of marbles, what is probability we identify all 10 classes of marbles after 100 draws?

## computational learning theory
- frameworks
  1. PAC
  2. mistake-bound - split into b processes which each fail with probability at most $\delta / b$
- questions
  1. *sample complexity* - how many training examples needed to converge
  2. *computational complexity* - how much computational effort needed to converge
  3. *mistake bound* - how many training examples will learner misclassify before converging
- must define convergence based on some probability

### PAC - probably learning an approximately correct hypothesis - Mitchell
- want to learn C
  - data X is sampled with Distribution D
  - learner L considers set H of possible hypotheses
- *true error* $err_d (h)$ of hypothesis h with respect to target concept c and distribution D is the probability that h will misclassify an instance drawn at random according to D.
  - $err_D(h) = \underset{x\in D}{Pr}[c(x) \neq h(x)]$
- getting $err_D(h)=0$ is infeasible
- *PAC learnable* - consider concept class C defined over set of instances X of length n and a learner L using hypothesis space H
  - C is PAC-learnable by L using H if for all $c \in C$, distributions D over X, $\epsilon$ s.t. 0 < $\epsilon$ < 1/2 $\delta$ s.t. $0<\delta<1/2$, learner L will with probability at least $(1-\delta)$ output a hypothesis $h \in H$ s.t $err_D(h) \leq \epsilon$
- *efficiently PAC learnable* - *time* that is polynomial in $1/\epsilon, 1/\delta, n, size(c )$
  - *probably* - probability of failure bounded by some constant $\delta$
  - *approximately correct* - err bounded by some constant $\epsilon$
  - assumes H contains hypothesis with artbitraily small error for every target concept in C

### sample complexity for finite hypothesis space - Mitchell
- *sample complexity* - growth in the number of training examples required
- *consistent learner* - outputs hypotheses that perfectly fit training data whenever possible
  - outputs a hypothesis belonging to the version space
- consider hypothesis space H, target concept c, instance distribution $\mathcal{D}$, training examples D of c. The versions space $VS_{H,D}$ is *$\epsilon$-exhaused* with respect to c and $\mathcal{D}$ if every hypothesis h in $VS_{H,D}$ has error less than $\epsilon$ with respect to c and $\mathcal{D}$: $(\forall h \in VS_{H,D}) err_\mathcal{D} (h) < \epsilon$

### rectangle learning game - Kearns
- data X is sampled with Distribution D
- simple soln: tightest-fit rectangle
- define region T so prob a draw misses T is $1-\epsilon /4$
  - then, m draws miss with $(1-\epsilon /4)^m$
    - choose m to satisfy $4(1-\epsilon/4)^m \leq \delta$

### VC dimension
- *VC dimension* measures *capacity* of a space of functions that can be learend by a statistical classification algorithm
  - let H be set of sets and C be a set
  - $H \cap C := \{ h \cap C \: \vert  h \in H \}$
  - a set C is *shattered* by H if $H \cap C$ contains all subsets of C
  - The VC dimension of $H$ is the largest integer $D$ such that there exists a set $C$ with cardinality $D$ that is shattered by $H$
- VC (Vapnic-Chervonenkis) dimension - if data is mapped into sufficiently high dimension, then samples will be linearly separable (N points, N-1 dims)
- VC dimension 0 -> hypothesis either always returns false or always returns true
- *Sauer's lemma* - let $d \geq 0, m \geq 1$, $H$ hypothesis space, VC-dim(H) = d. Then, $\Pi_H(m) \leq \phi (d,m)$
- fundamental theorem of learning theory provides bound of m that guarantees learning: $m \geq [\frac{4}{\epsilon} \cdot (d \cdot ln(\frac{12}{\epsilon}) + ln(\frac{2}{\delta}))]$

## concept learning and the general-to-specific ordering
- definitions
  - *concept learning* - acquiring the definition of a general category given a sample of positive and negative training examples of the category
    - concept is boolean function that returns true for specific things
    - can represent function as vector acceptable features, ?, or null (if any null, then entire vector is null)
  - *general hypothesis* - more generally true
    - general defines a partial ordering
  - a hypothesis is *consistent* with the training examples if it correctly classifies them
  - an example x *satisfies* a hypothesis h if h(x) = 1
- *find-S* - finding a maximally specific hypothesis
   - start with most specific possible
    - generalize each time it fails to cover an observed positive training example
    - flaws
       - ignores negative examples
    - if training data is perfect, then will get answer
       1. no errors
        2. there exists a hypothesis in H that describes target concept c
- *version space* - set of all hypotheses consistent with the training examples
  - *list-then-eliminate* - list all hypotheses and eliminate any that are inconsistent (slow)
  - *candidate-elimination* - represent most general (G) and specific (S) members of version space
    - version space representation theorem - version space can be found from most general / specific version space members
    - for positive examples
      - make S more general
      - fix G
    - for negative examples
      - fix S
      - make G more specific
    - in general, optimal query strategy is to generate instances that satisfy exactly half the hypotheses in the current version space
  - testing?
    - classify as positive if satisfies S
    - classify as negative if doesn't satisfy G
- inductive bias of candidate-elimination - target concept c is contained in H