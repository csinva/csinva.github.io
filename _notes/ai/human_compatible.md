---
layout: notes
title: human compatible
category: ai
---

*A set of notes based on the book human compatible, by Stuart Russell*

# what if we succeed?

- candidates for biggest event in the future of humanity
  - we all die
  - we all live forever
  - we conquer the universe
  - we are visited by a superior alien civilization
  - we invent superintelligent AI
- *defn*: humans are intelligent to the extent that our actions can be expected to achieve our objectives (given what we perceive)
  - machines are *beneficial* to the extent that *their* actions can be expected to achieve *our* objectives
- Baldwin effect - learning can make evolution easier
- **utility** for things like money is *diminishing*
  - rational agents maximize **expected utility**
- McCarthy helped usher in *knowledge-based systems*, which use *first-order logic*
  - however, these didn't incorporate uncertainty
  - modern AI uses utilities and probabilities instead of goals and logic
  - bayesian networks are like probabilistic propositional logic, along with bayesian logic, probabilistic programming languages
- language already encodes a great deal about what we know
- *inductive logic programming* - propose new concepts and definitions in order to identify theories that are both accurate and concise
- want to be able to learn many useful abstractions
- a superhuman ai could do a lot
  - e.g. help with evacuating by individually guiding every person/vehicle
  - carry out experiments and compare against all existing results easily
  - high-level goal: raise the standard of living for everyone everywhere?
  - AI tutoring
- EU GDPR's "right to an explanation" wording is actually much weaker: "meaningful information about the logic involved, as well as the significance and the envisaged consequences of such processing for the data subject."
- whataboutery - a method for deflecting questions where one always asks "what about X?" rather than engaging

## harms of ai

- ex. surveillance, persuasion, and control
- ex. lethal autonomous weapons (these are scalable)
- ex. automated blackmail
- ex. deepfakes / fake media
- ex. automation - how to solve this? Universal basic income?



## value alignment

- ex. king midas
- ex. driving dangerously
- ex. in optimizing sea oxygen levels, takes them out of the air
- ex. in curing cancer, gives everyone tumors
- note: for an AI, it might be easier to convince of a different objective than actually solve the objective
- basically any optimization objective will lead AI to disable its own off-switch



## possible solns

- Oracle AI - can only answer yes/no/probabilistic questions,  otherwise no output to the real world
- inverse RL
  - ai should be uncertain about utitilies
  - utilties should be inferred from human preferences
  - in systems that interact, need to express preferences in terms of game theory
- complications
  - can be difficult to parse human instruction into preferences
  - people are different
  - AI loyal to one person might harm others
  - ai ethics
    - consequentalism - choices should be judged according to expected consequences
    - deontological ethics, vritue ethics - concerned with the moral character of actions + individuals
    - hard to compare utilties across people
    - utilitarianism has issues when there is negative utility
  - preferences can change
- AI should be regulated
- deep learning is a lot like our sensory systems - logic is still need to act on these abstractions