---
layout: notes_without_title
section-type: notes
title: complexity
category: research
---

[TOC]

# philosophy

- [What is complexity?](http://cogprints.org/357/4/evolcomp.pdf) (edmonds 95)
  - complexity is only really useful for comparisons
  - properties
    - size - makes things potentially complex
    - ignorance - complex things represent things we don't understand (e.g. the brain)
    - minimum description size - more about **information** than complexity
      - potential problem - expressions are not much more complex than the original axioms in the system, even though they can get quite complex
      - potential problem - things with lots of useless info would seem more complex
    - some variety is necessary but not sufficient
    - order - we sometimes find order comes and goes (like double descent) - has a lot to do with language (audience)
  - defn: "that property of a language expression which makes it difficult to formulate its overall behaviour even when given almost complete information about its atomic components and their inter-relations"
    - language matters - what about the system are we describing?
    - goal matters - what outcome are interested in?
- [On Complexity and Emergence](https://arxiv.org/abs/nlin/0101006) (standish 01)
  - definition close to Kolmogorov / shannon entropy
  - adds context dependence
  - kolmogorov complexity = algorithmic information complexity
    - problem 1: must assume a particular Universal Turing machine (which might give differing results)
    - problem 2: random sequences have max complexity, even though they contain no information
  - soln
    - incorporate context - what descriptions are the same?
    - $C(x) = \lim _{\ell \to \infty} \log_2 N - \log_2 \omega (\ell, x)$
      - where C(x) is the complexity (measured in bits), ℓ(x) the length of the description, N the size of the alphabet used to encode the description and ω(ℓ,x) the size of the class of all descriptions of length less than ℓ equivalent to x.
  - emergence - ex. game of life
- [What is complexity 2](https://link.springer.com/chapter/10.1007/978-3-642-50007-7_2) (Gell-Mann 02)
  - AIC - algorithmic information content - contains 2 terms
    - effective complexity (EC) = the length of a very concise description of an entity's regularities
      - regularities are judged subjectively (e.g. birds would judge a bird song's regularity)
    - 2nd term relating to random features
- [complexity](http://www.scholarpedia.org/article/Complexity) (Sporns 07)
  - complexity = degree to which **components** engage in organized structured **interactions**
  - High complexity -> mixture of order and disorder (randomness and regularity) + have a high capacity to generate **emergent** phenomena.
  - (simon 1981): complex systems are “made up of a large number of parts that have many interactions.”
  - 2 categories
    - algorithmic / mdl
    - natural complexity (e.g. physical complexity)
      - ![300px-Complexity_figure1](assets/300px-Complexity_figure1.jpg)
- [quanta article](https://www.quantamagazine.org/computer-science-and-biology-explore-algorithmic-evolution-20181129/?fbclid=IwAR0rSImplo7lLM0kEYHrHttx8qUimB-482dI9IFxY6dvx0CFeEIqzGuir_w)
  - "the probability of producing some types of outputs is far greater when randomness operates at the level of the program describing it rather than at the level of the output itself"
  - "they [recently reported in *Royal Society Open Science*](http://rsos.royalsocietypublishing.org/content/5/8/180399) that, compared to statistically random mutations, this mutational bias caused the networks to evolve toward solutions significantly faster."

# minimum description length

- [mdl intro](http://www.scholarpedia.org/article/Minimum_description_length)
  - coding just the data would be like maximum likelihood
  - minimize $\underset{\text{log-likelihood}}{-\log P(y^n|x^n;\theta)} + \underset{\text{description length}}{L(\theta)}$
    - ex. OLS
    - if we want to send all the coefficients, assume an order and $L(\theta) = L(p) + L(\theta_1, ... \theta_p)$
      - $L(\theta) \approx \frac p 2 \log p$
        - quantization for each parameter (must quantize otherwise need to specify infinite bits of precision)
    - if we want only a subset of the coefficients, also need to send $L(i_1, ..., i_k)$ for the indexes of the non-zero coefficients
  - minimization becomes $\underset p \min \quad [\underset{\text{noise}}{- \log P(y^n|x^n; \hat{\theta}_{OLS})} + \underset{\text{learnable info}}{(p/2) \log n}]$
    - *noise* - no more info can be extracted with this class of models
    - *learnable info* in the data = precisely the best model
    - **stochastic complexity** = *noise* + *learnable info*
    - in this case, is same as BIC but often different
  - modern mdl - don't assume a model form, try to code the data as short as possible with a *universal* model class
    - often can actually construct these codes
- Kolmogorov complexity K(x) = the shortest computer program (in binary) that generates x (a binary string) = the "amount of info" in x
  - complexity of a string x is at most its length
  - algorithmically random - any string whose length is close to $|x|$
    - more random = higher complexity
- Minimum description length original reference \cite{rissanen1978modeling}. What is the minimum length description of the original?
  - MDL reviews \cite{barron1998minimum, hansen2001model}.
  - Book on stochastic complexity \cite{rissanen1989stochastic}
  - *Minimum Description Length*, *MDL*, principle for model selection, of which the original form states that the best model is the one which permits the shortest encoding of the data and the model itself
- *note: this type of complexity applies to the description, not the system*

# computational complexity

- amount of computational resource that it takes to solve a class of problem
- [Computational complexity](https://dl.acm.org/citation.cfm?id=1074233) (Papadimitriou)
  - like run times of algorithms etc. $O(n)$
- [Parameterized complexity](https://www.researchgate.net/profile/Michael_Fellows/publication/2376092_Parameterized_Complexity/links/5419e9240cf25ebee98883da/Parameterized-Complexity.pdf) (Downey and Fellows)
  - want to solve problems that are NP-hard or worse, so we isolate input into a parameter

# bayesian model complexity

- [Bayesian measures of model complexity and fit](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/1467-9868.00353) (spiegelhalter et al.)
- AIC
- BIC

# statistical learning theory

- VC-dimension - measure of capacity of a function class that can be learned
  - cardinality of largest number of points which can be shattered

# misc

- [rashomon curves](https://arxiv.org/abs/1908.01755) (semenova & rudin, 2019)
  - **rashomon effect** - many different explanations exist for same phenomenon
  - **rashomon set** - set of almost-equally accurate models for a given problem
  - **rashomon ratio** - ratio of volume of set of accurate models to the volume of the hypothesis space
  - **rashomon curve** - empirical risk vs rashomon ratio
    - **rashomon elbow** - maximize rashomon ratio while minimizing risk
      - good for model selection
- bennet's logical depth (1988) - computational resources taken to calculate the results of a minimal length problem (combines computational complexity w/ kolmogorov complexity)
- Effective measure complexity (Grassberger, 1986) quantifies the complexity of a sequence by the amount of information contained in a given part of the sequence that is needed to predict the next symbol
- Thermodynamic depth (Lloyd and Pagels, 1988) relates the entropy of a system to the number of possible historical paths that led to its observed state
- lofgren's interpretation and descriptive complexity
  - convert between system and description
- kaffman's number of conflicting constraints
- Effective complexity (Gell-Mann, 1995) measures the minimal description length of a system’s regularities
- Physical complexity (Adami and Cerf, 2000) is related to effective complexity and is designed to estimate the complexity of any sequence of symbols that is about a physical world or environment
- Statistical complexity (Crutchfield and Young, 1989) is a component of a broader theoretic framework known as computational mechanics, and can be calculated directly from empirical data
- Neural complexity (Tononi et al., 1994) - multivariate extension of mutual information that estimates the total amount of statistical structure within an arbitrarily large system.= the difference between the sum of the component’s individual entropies and the joint entropy of the system as a whole



# todo 

- add bracket entropy kind of thing