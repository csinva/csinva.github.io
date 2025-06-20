---
layout: notes
title: philosophy
category: ai
subtitle: Notes on philosophy relevant to explanation (particularly in science)
---


{:toc}

# basics

- try to understand what principles underly all phenomena

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



# the beggining of infinity (david deutsch, 2011)

- all progress has resulted from the quest for good explanations
  - *good explanation* - hard to vary while still accounting for what it purports to account for
  - the *reach* of an explanation is something we find out only after the fact (though we strive for explanations with good reach)
  - seek "an idea so simple...that...we sill all say to each other, how could it have been otherwise"?
- tenets
  - problems are inevitable
  - problems are soluble
  - optimisim = the proposition that all evils are due to a lack of knowledge, and that knowledge is attainable by the methods of reason and science
    - e.g. the distence between a 'natural' disaster andone brought about by ignorance is parochial (e.g. famine)
- theories
  - empiricism - we "derive" all our knowledge from sensory experience (deutsch argues that evidence is not used to generate theories but rather to choose between theories that have already been guessed)
  - inductivisim - idea that theories are obtained by generalizing repeated experiences
  - instrumentalism - science cannot describe reality, only predict outcomes of observations
- evolution
  - evolution (darwinian) - creation of knowledge through alternating variation & selection
  - lamarckism - mistaken evolutionary theory that adaptations are acquired by organisms during their lifetime then passed down
  - fine-tuning - if the constants or laws of physics were slightly different, there would be no life
  - meme - explanations must spread both by being replicated and enacted in the brains of others
  - a substantial proportion of all evolution on our planet to date has occured in human brains
  - only 2 basic strategies of meme replication
    - help holders (rational)
    - distable holder's critical faculties (anti-rational)
  - creativity developed as a way to better emulate ordinary things for mate selection, but proved to have much greater reach
- morality
  - moral philosophy addresses the problem of what sort of life to want
  - reductionism - misconception that we must always explain things by analyzing them into components
  - 'you can't derive an ought from an is' (paraphrase from David Hume)
- politics
  - just as science seeks explanations that are experimentally testable, a rational political system makes it as easy as possible to detect + persuade others that a leader or policy is bad, and to remove them without violence if they are (popper's criterion)
  - the moral imperative not to destroy the means of correcting mistakes could be the only moral imperative
  - arguments against the problem of apportioning representatives -- argues instead for plularity voting system
  - a culture is a set of ideas that cause their holders to behave alike in some ways
- aesthetics
  - when a piece of music has the attribute 'displace one note and there would be diminishment', there is an explanation (one day it may be expressible in words)
- jump to universality - gradually improving systems undergo a sudden large increase in functionality
  - e.g. numbering systems failed a couple times before becoming universal
  - e.g. genetic code
  - many theories for sustaining may be too limiting (e.g. propose studying carbon capture rather than limiting carbon production)
- minor
  - people in 1900 did not consider the internet or nuclear power unlikely: they did not conceive of them at all
  - the original sources of scientific theories are almost never good - weird that philiosophy places such an emphasis on original texts
  - most people believe that an income of about twice their own should be sufficient to satisfy any reasonable person (hard to imagine what it would be like to have twice as much) - David Friedman

# philosophy of science

## [thomas kuhn](https://plato.stanford.edu/entries/thomas-kuhn/)

- science enjoys periods of stable growth punctuated by revisionary revolutions
  - the development of science is driven, in normal periods of science, by adherence to what Kuhn called a ‘paradigm’
    - The functions of a paradigm are to supply puzzles for scientists to solve and to provide the tools for their solution
  - A crisis in science arises when confidence is lost in the ability of the paradigm to solve particularly worrying puzzles called ‘anomalies’
    - scientific revolutions involve a revision to existing scientific belief or practice
- ‘incommensurability thesis’ -  theories from differing periods suffer from certain deep kinds of failure of comparability
  - controversial - goes against the idea that science constantly builds

# the story of philosophy

**book by will durant**

- states his own view that philosophy should focus on ethics rather the epistemiology (i.e. how we know what we know)
  - every science begins as philosophy and ends as art
- some definitions of philosophy
  - pursuit of fundamental laws
  - quest of unity
- plato
  - socrates, plato's teacher pursued stricter definitions and was put to death
  - plato writes *the Republic* - fictitional dialogue w/ socrates as the protagonist
    - argues that democracy failes because people are greedy
    - advocates for an absolute meritocracy with 3 classes
      - ruling class should live like communists, decent state salary, disallowed from excess
      - soldiers / auxiliaries
      - general population
    - requires equality of education
    - requires religion to placated the non-ruling majority
    - excess is regulated
    - justice = having and doing what is one's own
      - each shall receive equivalent to what he produces + perform function for which he is best fit
    - juxtapositions
      - jesus: kindness to the weak
      - nietzsche: bravery of the strong
      - plato: effective harmony of the whole
  - only 3 things: truth, beauty, + justice
- aristotle
  - starts systematic science, library science, and **logic**
  - advocates for *uinversals* as individuals (e.g. a man, not *man* like Plat argues for)
    - this is more grounded in reality
  - theology: God moves the work like force, but does little else
  - science: infinitesimal distinctions - boundaries between plant/animal categories are blurry
    - form: man, matter: child = possibility of form
  - politics: ideally a monarchy / aristocracy but more realistically would be constitutional gov. (people determine needs, leaders determine how to meet them)
    - restrictions on pop.
    - believes in slavery / female inferiority
- francis bacon
  - lived in 1500s/1600s in England
  - father of the scientific method
    - objective and realistic
    - in contrast to descartes = subjctive/idealistic
      - "I think therefore I am"
  - bacon embraces **epicureanism** - don't want anything
  - scors knowledge that doesn't lead to action
  - **science** = organization of knowledge
  - **philosophy** = organization of science
  - doubt all assumptions
- spinoza
  - baroch de espinoza
  - jew who was excommunicated for anti-religious writing
  - no distinction between body and mind
  - no free will - only desires that guide everything
	  - beginnings of doubting rationalism
- voltaire
  - frenchman who was exiled
  - seeks history of ideas, beginning with *The Essay on Morals*
  - *Candide* - short story, denouncing optimism for pragmatism
  - real philosophy begins with *Philosophic dictionary*
  - strongly against superstition
  - wrote simple, accessible pamphlets
  - in his later years, turns to focus on the pursuit of usefulness rather than truth
  - contrasts with younger Roussea, who wanted more action, instinct, social contract
- kant: mind has prior beliefes
  - what makes a math law better than some other thing? kant says a priori beliefs...interestingly those beliefs were from evolution in the first place
  - mind is not blank slate: mind filters in what we perceive *a priori* in contrast to growing popular belief that everything comes from perception
  - understanding can never go beyond the limits of sensibility
    - certain things in science/religion etc. can never be known, just interpreted
    - time and space are not realities but just our interpretations
  - lots of connections to priors in modern AI research
  - morals come from an innate sense
  - somewhat pro-religion but not fully, still still faced persecution in Prussia
  - "Have strongly-held values, and malleable opinions". - Francois Chollet tweet
- schopenhauer
  - everything is will: continuing trend from espinoza + kant against rationalism
  - pessimist: even in Utopia, ennui sets in
  - objects of science is universal that contains many particulars while object of art is particular than contains a universal -- this requires more genius
- herbert spencer
  - evolution as a guiding philosophy of everything
  - darwin published *Origin of Species* in 1859, when spencer was 40
    - spencer is thus more lamarckian
  - greatest contributions were to sociology: carefully curates data for sociology analysis
  - resulting philosophy is conservative, laissez-fare, anti-regulation
- friedrich nietzsche
  - evolution as morality: favors the strong
  - germans have 2 words for good / bad - one is closer to strong, the other to kind
  - everything is due to an underlying will for power
  - evolution towards "the superman"
- bertrand russel
  - starts with symbolic reasoning
  - after WWI, shifts tow grounded philosophy in pacifism, communism

# stability

- [Foundationalism](http://en.wikipedia.org/wiki/Foundationalism) - where the chain of justifications eventually relies on [basic beliefs](http://en.wikipedia.org/wiki/Basic_beliefs) or [axioms](http://en.wikipedia.org/wiki/Axiom) that are left unproven
  - Plato’s Republic
- the stability of belief: how rational belief coheres with probability (leitgeb, 2017) - introduction
- [To Explain or to Predict](https://projecteuclid.org/download/pdfview_1/euclid.ss/1294167961)? (Shmueli, 2010)
  - explanatory modeling as the use of statistical models for testing causal explanations
  - many philosophies view explanation and prediction as distinct (but not incompatible)
  - 4 major aspects: causation-association, theory-data, retrospective-prospective, bias-variance
  - (causal) explanation is often more about picking the right class of models (which minimizes bias) rather than fitting  their parameters

