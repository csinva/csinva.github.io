---
layout: notes_without_title
section-type: notes
title: complexity
category: research
---

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
  -

# minimum description length

- Kolmogorov complexity K(x) = the shortest computer program (in binary) that generates x (a binary string).
  - complexity of a string x is at most its length
  - algorithmically random - any string whose length is close to |x|
- Minimum description length original reference \cite{rissanen1978modeling}. What is the minimum length description of the original?
  - MDL reviews \cite{barron1998minimum, hansen2001model}.
  - Book on stochastic complexity \cite{rissanen1989stochastic}
  - *Minimum Description Length*, *MDL*, principle for model selection, of which the original form states that the best model is the one which permits the shortest encoding of the data and the model itself
- [mdl intro](http://www.scholarpedia.org/article/Minimum_description_length)
  - coding just the data would be like maximum likelihood
  - minimize $\underset{\text{log-likelihood}}{-\log P(y^n|x^n;\theta)} + \underset{\text{description length}}{L(\theta)}$
    - if we want to send all the coefficients, assume an order and $L(\theta) = L(p) + L(\theta_1, ... \theta_p)$
      - $L(\theta) \approx \frac p 2 \log p$
        - quantization for each parameter

# computational complexity

- amount of computational resource that it takes to solve a class of problem
- [Computational complexity](https://dl.acm.org/citation.cfm?id=1074233) (Papadimitriou)
  - like run times of algorithms etc. $O(n)$
- [Parameterized complexity](https://www.researchgate.net/profile/Michael_Fellows/publication/2376092_Parameterized_Complexity/links/5419e9240cf25ebee98883da/Parameterized-Complexity.pdf) (Downey and Fellows)
  - want to solve problems that are NP-hard or worse, so we isolate input into a parameter

# misc

- bennet's logical depth - computational resources taken to calculate the results of a minimal length problem (combines computational complexity w/ kolmogorov complexity)
- lofgren's interpretation and descriptive complexity
  - convert between system and description
- kaffman's number of conflicting constraints

# bayesian model complexity

- [Bayesian measures of model complexity and fit](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/1467-9868.00353) (spiegelhalter et al.)
- AIC
- BIC

# statistical learning theory

- VC-dimension - measure of capacity of a function class that can be learned
  - cardinality of largest number of points which can be shattered
