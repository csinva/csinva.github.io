# Early Vision in Artificial Neural Networks

**Chris Olah, OpenAI Clarity**

- dnn is a good model for brain, but we can do much more
- 3 claims
  - features are the fundamental unit of dnns - they correspond to directions
    - 4 probes
      - max activation
      - dataset example
      - tuning curve (relative to rotated dataset ex which maximally activate a neuron)
      - response to synthetic stimuli
    - ex. curves, high-low freq detetctors, but some neurons respond to multiple things
      - can use all probes on these
      - can do this for high-level features as well
  - feature are connected by weights, forming circuits
    - can look at weights from simple curves to full curves
      - makes sense (same weights rotated) and opposite orientations negative
    - ex. oriented dog detector
      - one pathway for left oriented dog
      - another for right oriented dog
      - they get added later
    - general theme - inhibition when there are features that are similar enough to be confused
    - ex. superposition (polysemanticity) - this makes it difficult to understand
      - learn a car detector
      - then stash it into a bunch of dog detectors later in the network
    - can we connect this to adversarial examples?
    - can remove some connections and rerun max activation to see how it changes
  - analagous features and circuits form across models and tasks
    - ex. see curves across models
      - curves seem to also look like perpendicular lines to the edges
  - weight max activation images by their importance scores?
    - l2 reg does smth similar
    - look at inflection points of neuron tuning curves, not max