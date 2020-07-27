# priors for semantic variables

**yoshua bengio**

- biggest problem: OOD generalization
- key to overcoming this: compositionality
  - ex. distributed representations, composition of layers
- systematic generalization in language, analogies, abstract reasoning?
  - dynamically recombine existing concepts
  - current methods over fit the training **distr** - with more training data do worse on OOD
- system 1 (fast, unconscious) vs system 2 (slow, conscious)
  - implicit vs verbalizable knowledge
  - would be nice to have subsystems that could handle these separately
- system 2 inductive priors
  - sparse factor graph (bengio 2017)
    - not marginal independence, e.g. ball and hand
    - **ex. independent latents in GAN (e.g. z) are not meaningful, but later down (e.g. style space) are meaningful** - this is bc independence is too strong a constraint
  - semantic variables are causal
    - thus, changing one variable should change others
    - having the right causal structure usually helps you adapt faster
  - distr. changes due to localized causal interventions
  - shared rules across instance tuples
  - meaning (e.g. in encoder) is stable + robust wrt distr changes
  - credit assignment is only over short causal chains
  - concepts can be mapped to words / languages
- attention is useful
- ex. recurrent independent mechanisms - combine different modules with attention + bottlenecks
- learning to combine top-down and bottom-up feedback