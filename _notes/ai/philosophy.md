---
layout: notes
title: philosophy
category: ai
---

**Notes on philosophy relevant to explanation (particularly in science)**


{:toc}

# interpretable ml

- [Machine Learning and the Future of Realism](https://arxiv.org/pdf/1704.04688.pdf) (hooker & hooker, 2017)
  - lack of interpretability in DNNs is part of what makes them powerful
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
- [Explainable Artificial Intelligence and Machine Learning: A reality rooted perspective](https://arxiv.org/pdf/2001.09464v1.pdf) (Emmert-Streib et al. 2020)
  - explainable AI is not a new field but has been already recognized and discussed for expert systems in the 1980s
  1. in some cases, such as simple physics, we can hope to get a **theory** - however, when the underlying process is complicated, interpretation can't hope to simplify it
  2. in other cases, we might hope just for a **description**

# [scientific explanation](https://plato.stanford.edu/entries/scientific-explanation/) (SEP)

- distinction between scientific explanation and non-scientific explanation (although can put things on a spectrum)
- distinction between explanations and accounts which are "merely descriptive"
- these theories focus only on explanations of *why* things happen
- concepts: explanation, description, causation

## some overarching examples - what causes what?

- do birth pills stop pregnancy (both for men/women)?
- sun, flagpole, shadow
- barometer, air pressure, storm
- collision of pool balls
- supply/demand curves in economics
- gas law $PV=nRT$

## DN model = Deductive-Nomological Model

- by Popper, Hempel, Oppenheim (popper 1935, hempel 1942)
- explanation has 2 constituents
  - **explanandum** - sentence describing phemonenon to be explained
  - **explanans** - class of sentences used to account for the phenomenon
- deductive
  - explanans must be true
  - explanandum must be a logical consequence of the explanans
- nomological = "lawful"
  - explanans must contain a "law of nature"
  - the law of nature must be essential to deriving the explanans
- still hard to decide exactly what is a law - should be exceptionless generalizations describing regularities
  - can have probabilistic laws as well (e.g. prob of recovering after taking penicillin is high)
- why this framework
  - by framing terms of laws and cricumstances, the explanation shows that the phenomenon *was to be expected* and helps us *understand why* it occured
  - sometimes, an *explanation-sketch* uses words like cause, that can be reframed more precisely in the DN model
- counterexamples
  - assymetry (e.g. shadow length, flagpole height, sun angle) - derive shadow length seems explanatory, but not deriving flapole height
  - irrelevant details (e.g. being a man + taking birth control pills explains why you don't get pregnant)
  - feels like their is something missing about "causality", but this is difficult to pin down - suggests DN model states necessary but not sufficient conditions
  - features of an explanation must be recognized / used by users of an explanation

## statistical relevance - wesley salmon

- starts with (salmon, 1971)
- given some class or population *A*, an attribute *C* will be **statistically relevant** to another attribute *B* iff $P(B∣A,C) \neq P(B∣A)$
  - find a set of attributes which divide the target into a homogenous partition = even if we split the cells further, they keep the same probability
    - like in causal inference, assume no missing variables
  - then, the explanation for a new target *x* gives the cells, the prob. for each of the cells, and which cell *x* belongs to
- this method is almost information-theoretic - same explanans equally explains the same model with inverted probabilies
- ex. (Salmon, 1971) atmospheric pressure, barometer, and storm - barometer is explanatory but not causal

## causal mechanical models

- starts with (salmon, 1984)
- elements
  - **causal process** - leaves marks on the world which persist spatiotemporally (these marks hint at counterfactuals)
    - contrasts with a pseudoprocess, like a shadow
  - **causal interaction** - interaction of such processes, e.g. a car crash
- explanation consists of 2 parts which both track causal process + interactions
  - **etiological** - leading up to event
  - **constitutive parts** - during event
- issues: hard to distinguish between explanatory causes (e.g. for a pool ball collision, mass + velocity) vs other so-called causal processes (e.g. chalk mark on ball)
  - tends towards overly complex physical descriptions - e.g. for gas law track all particles rather than something global like pressure



## unificationist models

- important attempts include friedman (1974) and kitcher (1989)
- seeks a unified account of a range of different phenomena
- best explanations explains most phenomena with as few + as stringent arguments possible
- potential issues
  - causal asymmetries - equally likely to say planets in future cause motion now then planets now cause motion in the future - this is honestly probably fine
  - doesn't easily admit laws at different graunlarities

## pragmatic = contextual explanation models

- scriven (1962), bromberger (1966), van Fraassen (1980), achinstein (1983)
- takes audience into account
- others have been after characterizing a single "true" explanation and the role of the audience was minimized
- "pragmatic" here means not just useful but also explicitly considering psychology + context
- "Causal–explanatory pluralism" (lombrozo 2010 [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0010028510000253)]) - subjects prefer explanations that appeal to relationships that are relatively stable (in the sense of continuing to hold across changing circumstances)
- **constructive empiricism** (Bas van Fraasen)
  - just want theories that are "empirically adequate"
  - explanations are answers to questions - usually **why?** questions
    - explanations pick something from a set and tell why it is not any of the other things in the set (the set is given from context)
  - explanation is ..."a t three-term relation between theory, fact, and context"
  - asymmetries coem of rom context
- main criticism: this theory is too flexible, ironically it might be too flexible to be meaningfully (practically) useful
  - however, it is possible that this is the most flexible framework that is still accurate
  - want context to come in for as few steps as possible during an explanation - maybe we don't need to analyze human psych
  - somewhat circular - if this is true, then how can we resolve ambiguities?



## open areas

- understand casuality
- more focus on whethery expalanations capture our intuitive judgements and more on the issue of why the info they convey is valuable + relates to our goals
- to what extent does a single model work across sciences (e.g. biologists claim to be interested in mechanisms whereas physicists in laws)

# misc

- [Foundationalism](http://en.wikipedia.org/wiki/Foundationalism) - where the chain of justifications eventually relies on [basic beliefs](http://en.wikipedia.org/wiki/Basic_beliefs) or [axioms](http://en.wikipedia.org/wiki/Axiom) that are left unproven
  - Plato’s Republic
- the stability of belief: how rational belief coheres with probability (leitgeb, 2017) - introduction
- https://projecteuclid.org/download/pdfview_1/euclid.ss/1294167961


# effective altruism

- [effectivealtruism](https://www.effectivealtruism.org/articles/introduction-to-effective-altruism/)
    - promising causes
      - Great in *scale* (it affects many lives, by a great amount)
      - Highly *neglected* (few other people are working on addressing the problem), and
      - Highly *solvable* or *tractable* (additional resources will do a great deal to address it).
    - 3 big areas
      - fighting extreme poverty (e.g. malaria)
      - animal suffering
      - improving the long-term future
- [rethink priorities jobs](https://rethinkpriorities.freshteam.com/jobs/iX8GfQ1eBLDq/researcher-multiple-positions-remote)
- [open philanthropy](https://www.openphilanthropy.org/blog/modeling-human-trajectory)
- careers can have greater impacts than altruism ([80k hours](https://80000hours.org/key-ideas/))
    - https://80000hours.org/career-guide/
    - [80k hours AI careers](https://80000hours.org/problem-profiles/positively-shaping-artificial-intelligence/)
- [givewell cost-effective calculations](https://www.givewell.org/how-we-work/our-criteria/cost-effectiveness)
    - charities often exaggerate their "cost to save a life"