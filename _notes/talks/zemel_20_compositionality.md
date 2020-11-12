# Discovering compositional representations with causal graphs and sketch drawings

**Richard Zemel, Vector Insitute (Toronto)**



- compositional causal representations
  - causal graphs
  - compositional representations (decompose into simpler pieces, or factors)
    - explicit - disentangled
    - implicit - transfer, generation (goal is not to explore the representation, but make it work better)
- sketches are nice - simple, intuitive, detail invariant
  - can be describes in a denser, vector graphics repr.
  - can we learn a model to produce good summary sketches?
  - trained a generative encoder-decoder model to mimic human sketches
  - learn sequential repr. using the path of the pen (assume it is piecewise linear)
- loss
  - loss for correct pen states, stroke supervision, and pixel agreement
- sketching learns disentangled space
  - vae embedding space usually separates things based on angles, displacements
  - sketchembedding not only separates them, but groups the clusters in a way that makes sense - entire manifold is learned - **compositional** - learns underlying components in a nested way
  - if we add/remove concepts in examples we can generate the desired compositional things
  - this embedding spaces helps downstream few-shot classification
- future work - try to make it be succinct - draw sketches that can classify a sketch as quickly as possible
  - can we learn sketches without stroke info?



## causal discovery in time-series

- amortized causal discovery - learning to infer causal graphs from time-series data

- how can you infer the influence of one brain region on another (e.g. fmri)?

  ![Screen Shot 2020-07-30 at 4.53.21 PM](assets/learning_causal_graphs.png)

- encoder outputs causal graph given a time series

  - decoder outputs the future of the time series, given the value + causal graph
  - like granger causality
  - decoder is shared between many time series, encoder changes between time-series

- key idea: train one model that can pool samples from different causal graphs

  - key assumption: this is possible assuming they share an underlying dynamics

