---
layout: notes
title: ai futures
category: ai
---

{:toc}

# human compatible
**A set of notes based on the book human compatible, by Stuart Russell 2019**

## what if we succeed?

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

# possible minds

**edited by John Brockman, 2019)**

## intro (brockman)

- new technologies = new perceptions
- we create tools and we mold ourselves through our use of them
- Wiener: "We must cease to kiss the whip that lashes us"
  - initial book *The human use of human beings*
  - he was mostly analog, fell out of fashion
  - initially inspired the field
- ai has gone down and up for a while
- gofai - good old-fashioned ai
- things people thought would be hard, like chess, were easy
- lots of physicists in this book...

## wrong but more relevant than ever (seth lloyd)

- current AI is way worse than people think it is
- wiener was very pessimistic - wwII / cold war
- singularity is not coming...



## the limitations of opaque learning machines (judea pearl)

- 3 levels of reasoning
  - statistical
  - causal
  - counterfactual - lots of counterfactuals but language is good and providing lots of them
- "explaining away" = "backwards blocking" in the conditioning literature
- starts causal inference, but doesn't work for large systems
- dl is more about speed than learning
- dl is not interpretable
- example: ask someone why they are divorced?
  - income, age, etc...
  - something about relationship...
- correlations, causes, explanations (moral/rational) - biologically biased towards this?
  - beliefs + desires cause actions
- randomly picking grants above some cutoff...
- pretty cool that different people do things because of norms (e.g. come to class at 4pm)
  - could you do this with ai?
- facebook chatbot ex.
- paperclip machine, ads on social media
- states/companies are like ais
- **equifinality** - perturb behavior (like use grayscale images instead of color) and they can still do it (like stability)

## the purpose put into the machine (stuart russell)

- want safety in ai - need to specify right objective with no uncertainty
- **value alignment** - putting in the right purpose
- ai research studies the ability to achieve objectives, not the design of those objectives
  - "better at making decisions - not making better decisions"
- want provable beneficial ai
- can't just maximize rewards - optimal solution is to control human to give more rewards
- cooperative inverse-rl - robot learns reward function from human
  - this way, uncertainty about rewards lets robot preserve its off-switch
  - human actions don't always reflect their true preferences

## the third law (george dyson)

- 2 eras: before/after digital computers
  - before: thomas hobbes, gottfried wilhelm leibniz
  - after: 
    - alan turing - intelligent machines
    - john von neumann - reproducing machines
    - claude shannon - communicate reliably
    - norbert weiner - when would machines take control
- analog computing - all about error corrections
- nature uses digitial coding for proteins but analog for brain
- social graphs can use digital code for analog computing
  - analog systems seem to control what they are mapping (e.g. decentralized traffic map)
- 3 laws of ai
  - ashby's law - any effective control system must be as complex as the system it controls
  - von neumman's law - defining characteristic of a complex system is that it constitutes its own simplest behavioral description
  - 3rd law - any system simple enough to be understandable will not be complicated enough to behave intelligently and vice versa

## what can we do? (daniel dennett)

- dennett wrote from bacteria to bach & back
- praise: willingness to admit he is wrong / stay levelheaded
- rereading stuff opens new doors
- import to treat AI as tools - real danger is humans being slaves to the AI coming about naturally
  - analogy to our dependence on fruit for vitamin C whereas other animals synthesize it
  - tech has made it easy to tamper with evidence etc.
  - Wiener: "In the long run, there is no distinction between arming ourselves and arming our enemies."
- current AI is parasitic on human intelligence
- we are robots made of robots made of robots...with no magical ingredients thrown in along the way
- current humanoid embellishments are *false advertising*
- need a way to test safety/interpretability of systems, maybe with human judges
- people automatically personify things
- we need intelligent tools, not conscious ones - more like oracles
- very hard to build in morality into ais - even death might not seem bad

## the unity of intelligence (frank wilczek)

- can an ai be conscious/creative/evil?
- mind is emergent property of matter $\implies$ all intelligence is machine intelligence
- david hume: 'reason is, and ought only to be, the slave of the passions'
- no sharp divide between natural and artificial intelligence: seem to work on the same physics
- intelligence seems to be an emergent behavior
- key differences between brains and computers: brains can self-repair, have higher connectivity, but lower efficiency overall
- most profound advantage of brain: connectivity and interactive development
- ais will be good at exploring
- defining general intelligence - maybe using language?
- earth's environment not great for ais
- ai could control world w/ just info, not just physical means
- affective economy - sale of emotions (like talking to starbucks barista)
- people seem to like to live in human world
  - ex. work in cafes, libraries, etc.
- future life institute - funded by elon...maybe just trying to make money

## lets aspire to more than making ourselves obsolete (max tegmark)

- sometimes listed as scaremonger
- maybe consciousness could be much more hype - like waking up from being drowsy
- survey of AI experts said 50% chance of general ai surpassing human intelligence by 2040-2050
- finding purpose if we aren't needed for anything?
- importance of keeping ai beneficial
- possible AIs will replace all jobs
- curiosity is dangerous
- 3 reasons ai danger is downplayed
    1. people downplay danger because it makes their research seem good - "It is difficult to get a man to understand something, when his salary depends on his not understanding it" - Upton Sinclair
    	- **luddite** - person opposoed to new technology or ways of working - stems from secret organization of english textile workers who protested
    2. it's an abstract threat
    3. it feels hopeless to think about
- AI safety research must precede AI developments
- the real risk with AGI isn't malice but competence
- intelligence = ability to accomplish complex goals
- how good are people at predicting the future of technology?
- joseph weizenbbam wrote psychotherapist bot that was pretty bad but scared him

## dissident messages (jaan taliin)

- voices that stand up slowly end up convincing people
- ai is different than tech that has come before - it can self-multiply
- human brain has caused lots of changes in the world - ai will be similar
- people seem to be tipping more towards the fact that the risk is large
- short-term risks: automation + bias
- one big risk: AI environmental risk: how to constrain ai to not render our environment uninhabitable for biological forms
- need to stop thinking of the world as a zero-sum game
- famous survery: katja grace at the future of humanity institute

## tech prophecy and the underappreciated causal power of ideas (steven pinker)

- "just as darwin made it possible for a thoughtful observer of the natural world to do without creationism, Turing and others made it possible for a thoughtful observer of the cognitive world to do without spiritualism"
- entropy view: ais is trying to stave off entropy by following specific goals
- ideas drive human history
- 2 possible demises
  - surveillance state
    - automatic speech recognition
    - pinker thinks this isn't a big deal because freedom of thought is driven by norms and institutions not tech
    - tech's biggest threat seems to be amplifying dubious voices not surpressing enlightened ones
    - more tech has correlated w/ more democracy
  - ai takes over
    - seems too much like technological determinism
    - intelligence is the ability to deploy novel means to attain a goal - doesn't specify what the goal is
    - knowledge are things we know - ours are mostly find food, mates, etc. machines will have other ones
- if humans are smart enough to make ai, they are smart enough to test it
- "threat isn't machine but what can be made of it"

## beyond reward and punishment (david deutsch)

- david deutsch - founder of quantum computing
- thinking - involves coming up w/ new hypotheses, not just being bayesian
- knowledge itself wasn't hugely evolutionarily beneficial in the beginning, but retaining cultural knowledge was
  - in the beginning, people didn't really learn - just remembered cultural norms
  - no one aspired to anything new
- so far, the way ais have been developed (e.g. chess-playing) is restricting a search space, but AGI wants them to come up with a new search space
- we usually don't follow laws because of punishments - neither will AGIs
- open society is the only stable kind
- will be hard to test / optimize for directly
- AGI could still be deterministic
- tension between imitation and learning? (immitation/innovation)
- people falsely believe AGI should be able to learn on its own, like Nietzche's *causa sui*, buy humans don't do this
- culture might make you more model-free

## the artificial use of human beings (tom griffiths)

- believes key to ml is human learning

- we now have good models of images/text, but not of 

- value alignment

- inverse rl: look at actions of intelligent agent, learn reward

- accuracy (heuristics) vs generalizability (often assumes rationality)
  - however, people are often not rational - people follow simple heuristics
  - ex. don't calculate probabilities, just try to remember examples

- people usually tradeoff time with how important a decision is - **bounded optimality**

- could ai actually produce more leisure?

## making the invisible visible (hans ulrich obrist)

- need to use art to better interpret visualizations, like deepdream
- ai as a tool, like photoshop
- tweaking simulations is art (again in a deep-dream like way)
- meta-objectives are important
- art - an early alarm system to think about the future, evocative
- design - has a clearer purpose, invisible
  - fluxist movement - do it yourself, like flash mob, spontanous, not snobby
- this progress exhibit - guggenheim where they hand you off to people getting older
- art - tracks what people appreciate over time
- everything except museums + pixels are pixels
- marcel duchamp 1917 - urinal in art museum was worth a ton

## algorists dream of objectivity (peter galison)

- science historian
- stories of dangerous technologies have been repeated (e.g. nanoscience, recombinant DNA)
- review in psychology found objective models outperformed groups of human clinicians ("prediction procedures: the clinical-statistical controversy")
- people initially started w/ drawing things
  - then shifted to more objective measures (e.g. microscope)
  - then slight shift away (e.g. humans outperformed algorithms at things)
- objectivity is not everything
- art w/ a nervous system
- animations with charcters that have goals

## the rights of machines (george church)

- machines should increasingly get rights as those of humans
- potential for AI to make humans smarter as well

## the artistic use of cybernetic beings (caroline jones)

- how to strech people beyond our simple, selfish parameters
- cybernetics seance art
- more grounded in hardware
- culture-based evolution
- uncanny valley - if things look too humanlike, we find them creepy
  - this doesn't happen for kids (until ~10 years)
- neil mendoza animal-based aft reflections
- is current ai more advanced than game of life?

---

## David Kaiser: Information for wiener, Shannon, and for Us

- wiener: society can only be understood based on analyzing messages
  - information = semantic information
  - shannon: information = entropy (not reduction in entropy?)
  - predictions
    - information can not be conserved (effective level of info will be perpetually advancing)
    - information is unsuited to being commodities
      - can easily be replicated
      - science started having citations in 17th century because before that people didn't want to publish
        - turned info into currency
      - art world has struggled w/ this
        - 80s: appropration art - only changed title
      - literature for a long time had no copyrights
      - algorithms hard to patent
  - wiener's warning: machines would dominate us only when individuals are the same
    - style and such become more similar as we are more connected
      - twitter would be the opposite of that
      - amazon could make things more homogenous
    - fashion changes consistently
      - maybe arbitrary way to identify in/out groups
    - comparison to markets
    - cities seem to increase diversity - more people to interact with
- dl should seek more semantic info not statistical info

## Neil Gershenfield: Scaling

- ai is more about scaling laws rathern that fashions
- mania: success to limited domains
- depression: failure to ill-posed problems
- knowledge vs information: which is in the world, which is in your head?
- problem 1: communication - important that knowledge can be replicated w/ no loss (shannon)
- problem 2: computation - import knowledge can be stored (von Neumann)
- problem 3: generalization - how to come up w/ rules for reasoning?
- next: fabrication - how to make things?
  - ex. body uses only 20 amino acids


