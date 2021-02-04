---
layout: notes
title: psychology
category: ai
---

Some notes on papers / books surrounding psychology, especially evolutionary psychology and the psychology of explanation. These are notes on the authors' points and not an endorsement of their views.

{:toc}

# explanations

## [The structure and function of explanations](https://www.sciencedirect.com/science/article/abs/pii/S1364661306002117) (lombrozo 2006)

- explanation structures
  - accommodate novel info in the context of prior beliefs
  - do so in a way that fosters generalization
- background
  - explanations answer *why* questions
  - cognitive science has embraced explanation with regard to concepts and prior knowledge
  - explanations affect: (1) prob. assigned to causal claims, (2) how properties are generalized, (3) learning
- predominant concepts
  - causation
  - pattern subsumption - this knowledge constrains what causes are probable/relevant
- function of explanations: predict/control the future, constraint for generalization
  1. causal inference - depends on prior beliefs + statistical evidence
    - explanations constrain causal inference based on prior beliefs
    - e.g. "if provided with evidence that cars of a particular color and size have better gas mileage, children and adults will disregard the confounding factor of color to conclude that car size causes the mileage difference"
    - people often offer explanations over evidence, especially when evidence is sparse
    - generating explanations for why a claim might be true provides a way to assess the probability of that claim in light of prior beliefs
    - "when generated from true beliefs, explanations provide an invaluable source of constraint; when generated from false beliefs, explanations can perpetuate inaccuracy."
  2. generalization of properties
    - basics of generalization
      - similarity: property is more likely to generalize to a new case if new case is similar
      - diversity: for generalizing to a broader category, property is more likely to generalize with more diverse evidence
    - explanations can override these basics
  3. generalization of knowledge systems
    - self-explanation aids learning
    - "Explaining to oneself thus facilitates generalization to transfer problems by isolating 
    - relevant senses of similarity, helping learners to overcome ‘the frailties of induction’"
- differences between explanation and causal reasoning
  
  - "some beliefs are privileged at the expense of others" - relevance determines which causal factors matter
  - "prior knowledge might not be deployed through other means" - e.g. "explaining why a claim might be true or false changes the perceived probability of that claim"
  - "properties of explanations, such as their generality or simplicity, can influence probabilistic judgements"

## Causal Explanation (lombrozo & vasilyeva 2017)

- explanations appeal to causes (although not all explanations are causal e.g. mathematics)
- causal inference here not the same as the way it is used in statistics

### causal *inference* w/ explanations

- "inference to the best explanation" - believe hypothesis that *best explains* the data
  - this is not just bayesian inference (a common assumption)
    - rather, it includes explanatory considerations such as simplicity, scope, and explanatory power
    - these things may improve short-term accuracy and make things easier to communicate/remember/use
    - it is possible that these things could be captured by a hierarchical bayesian model with appropriate priors / likelihoods
  - simplicity (lombrozo 2007) - explanation simplicity trades off w/ statistical likeliness
    - (lombrozo 2012) adults more likely to choose likeliness and all more like to choose likeliness for tasks with less apparent causal explanations
    - (pacer & lombrozo, 2017) explanation includes *node simplicity* = number of causes (nodes in graphical model) + *root simplicity* = number of unexplained causes (roots in graphical model)
      - people seem to only be sensitive to node simplicity
  - explanatory scope - how many things does this explanation imply (even if the others aren't tested)
    - e.g. does a diagnosis predict additional effects not yet tested? people prefer diagnoses with narrower scope (khemlani, sussman, and oppenheimer 2011)
  - explanatory power - people's explanations better predict their estimates of posterior probability than do objective probabilites on their own (douven & schupbach, 2015a, 2015b)
  - other considerations, such as coherence, completeness, manifest scope

### causal *discovery* w/ explanations

- causal discovery = causal model learning
- engaging in explanation influences causal model learning
  - being prompted to explain can promote understanding
  - makes them more likely to find underlying causal models
  - (walker et al. 2016) - children asked to explain more attune to both evidence and prior beliefs
  - also sometimes reinforces people's prior beliefs (right or wrong)
  - explaining “involved the integration of new information into existing knowl­edge” (Chi, De Leeuw, Chiu, and LaVancher, 1994)
- reasons why explanation alters causal learning
  - **attention** - explanation doesn't just boost attention - leads to specific benefits / deficits
  - **motivation** - explaining plays a motivational role  (e.g. gopnik 2000 "Explanation as orgasm")
    - i.e. people seeking good explanations motivates causal understanding
  - explanation favors finding hypotheses with "lovely" causes
  - however, some studies find that children prompted to explain outperform controls even when they don't generate the right explanation

### causal *responsibility* w/ explanations

- **causal responsibility** = to which cause(s) do we attribute a given effect?
  - ex. "why did she slip?" - either "she is clumsy!" or "the staircase is slippery!"
- classic ANOVA model (Kelley 1967) says ppl analyze covariation between factors such as person, stimulus, and situation but more seems to be involved
- different questions have different "contrast class" (van Fraasesen, 1980, philosophy) - why did *she* slip? vs why did she slip *on the stairs*?
  - different questions shift things to causal relevance and not just probability

## explanation taxonomy

- Aristotle’s 4 "causes" or modes of explanation

  | cause                                | description                                           | example                                          |
  | :----------------------------------- | ----------------------------------------------------- | ------------------------------------------------ |
  | **efficient**                        | proximal mechanisms of change                         | a carpenter is an efficient cause of a bookshelf |
  | **final** (functional, teleological) | the end, function or goal                             | holding books is a final cause of a bookshelf    |
  | **formal**                           | the form or properties that make something what it is | having shelves is a formal cause of a bookshelf  |
  | **material**                         | the substance of which something is constituted       | wood is a material cause of a bookshelf          |

- final causes
  - e.g. camouflage causes zebra stripes
  - real cause is a preceding intution
  - experiments suggest final causes are only accepted well when there is some causal link
    - e.g. adults who believe in God are more likely to accept scientifically unwarranted teleological explanations
- another taxonomy: inherent vs. extrinsic explanations (cimpian & salomon, 2014)
- formal explanations
  - pretty limited to category membership
  - e.g. Zach diagnoses ail­ ments because he is a doctor
  - these can be seen as **constitutive** (not causal): e.g. has four legs bc dog, but not is red bc is barn (even though most barns are read)
  - “existence of a whole presupposes the existence of its parts, and thus the existence of a part is rendered intelligible by identifying the whole of which it is a part” (prasada & dillingham 2009)
- people who gave different explanations (e.g. functional vs material) also generalized differently to different categories (lombrozo 09)
- two types of relationships - when bouth are present people opt for dependence
  - **dependence** = counterfactually - if cause didn't occur, effect wouldn't have occurred
  - **transference** = physical connection, e.g. continuous mechanism / conserved physical quantity

## causal mechanisms

- **mechanism** - spells out the intermediate steps between some cause and some effect
  - sometimes these are seen as explanations
- alternative define mechanisms as complex systems that involve a (typically hierarchi­cal) structure and arrangement of parts and processes, such as that exhibited by a watch, a cell, or a socioeconomic system
  - interlevel relationships are *constitutive*, not causal (e.g. saying molecules rub against one another - this *is* heat (contintutive) but people often misconstrue this is causal)
  - explanations can accomodate both types of relationships

## misc explanation work

- Evaluating computational models of explanation using human judgments (pacer, williams, chen, lombrozo, & griffiths, 2013) [[PDF](http://cocosci.princeton.edu/tom/papers/FormalExplanationModelUAI2013.pdf)]
  - overton 2012 finds that explanations used something general (ex. model) to explain something specific (ex. data)
  - subsequent analysis overton 2012 finds "inference to the best explanation" - use specific instances (ex. data) to draw general inferences
  - waskan et al 2014 - must be actually intelligible
  - lombrozo 2011 - explanations are intrinsically valuable, but also play an important instrumental role in the discovery and confirmation of intuitive theories, which in turn support prediction and intervention
  - explanations should be understood in terms of their role in generating understanding (Achinstein 1983; Wilkenfeld 2014), supporting future judgments (Craik 1943; Heider 1958; Quine & Ullian 1970), or **motivating the construction of causal theories (Gopnik 2000)**
  - explanations play a role in generalizing from known to novel cases (Rehder 2006; Sloman 1994; Lombrozo & Gwynne 2014)
  - sometimes impedes learning about properties that are idiosyncratic
  - explanatory errors and “illusions” can help us identify when and why engaging in explanation is so often beneficial
  - functional approach: why do we want explanations?
    - the best explanation for persuasion or efficient storage of information, for example, may not be the one that best supports future prediction.
  - evidence (=the explanandum) provides for some hypothesis (=the explanans).
- given causal thing what is best explanation
  - most relevant explanation model or explnatory tree model
  - human explanatory judgments track something more like evidence, information, or relevance, and not simply the prior or posterior probability of the explanans
- desiderata
  - **simplicity**
    - if simplicity does inform explanatory preferences, it is trumped or made moot by probability
    - count simplicity vs root simplicity (root is often preferred)
  - **fruitfulness**
    - generally explanations with broader scope are better except for causal stuff
- explanations + learning
  
  - explanation magnifies our prior beliefs
- "negative program" - empirical results disprove philosophical intuitions

# the invisible gorilla

- We think we experience much more of our physical world than we do
- We generally only see what we’re looking for
- Our memory is very fake
- We have a belief in shortcuts to expand our brain’s abilities
  - ex. Lumosity


# the moral animal

**Why We Are, the Way We Are: The New Science of Evolutionary Psychology** - Robert Wright, 1965. Notes in this section are not an endorsement of the author's views.

## sex, romance, and love

### darwin comes of age
- *Emile Durkheim* - father of modern sociology
- *Wilson* - initial book sociobiology was vehemently opposed
	- reactive against connotations of *social darwinism*
		- social darwinism is linked to eugenics
- genes affect human nature in two ways
	1. existence of guilt
	2. developmental program to calibrate guilt

### male and female
- have to consider environment of evolutionary adaptation (EEA)
- many studies on !Kung San of the Kalahari desert in Africa
- throughout nature, females are more coy while males are more promiscuous
	- this is true of every known human society
	- Samoa example was thought to be different, but this study was refuted
	- true with turkeys (who can be seduced by a wooden female turkey head)
- difference is due to amount of male parental investment
- apes
	- ex. gorilla alpha male claims all the women
	- ex. gibbons are monogamous - live in family units separate from others
		- sing duets

### gender differences in evolution

- ideas that humans are a *pair-bonding species*
	- humans require high male parental investment (MPI)
	- vulnerable offspring
	- more education from two parents
- genetically speaking, for males, worst thing is raising a child that isn't theres
	- experiments suggest they are most angry at sexual infidelity
	- historically, sometimes killed children that weren't theirs
	- quantity of sperm depends heavily on the amount of time a male's mate has been out of his sight lately
- genetically speaking, for females, worst thing is being abandoned
	
	- experiments suggest females are relatively more angry at emotional infidelity
	- however, can still be genetically useful for a female to be unfaithful
		1. can extract gifts for sex - "resource extraction"
		2. they don't advertise their ovulation - "seeds of confusion"
	
- *madonna-whore dichotomy* - a psychological phenomenon which groups females into 2 categories: marriage / fling

  - perhaps example of *frequency-dependent selection*

  - ex. blugill sunfish 
    1. normal males make nests and guard eggs
    2. drifter males sneak around and fertilize others' eggs
    - nature strikes a balance between both
  - in actuality, should be able to guage situation and switch between different behaviors
  - *self-esteem* might be biological marker that helps with this

- males are more likely to gain from leaving a marriage
	
	- females only have ~25 fertile years

### the marriage market
- polygamy, initially, seems to benefit males
	- one male can get more females
- why monogamy
	1. *ecologically imposed* - if people are struggling to survive, a woman shouldn't share a man with another
		- they won't have enough resources
	2. *socially imposed* - in economic unequal societies (like today)
		- these societies are the ones that have dowry
		- imagine 1000 men and 1000 women -> polygyny actually hurts the men
		- therefore, monogamy likely evolved to stop the dangers of men without wives
- current divorce rates are high, hurting all alike
- Charles Darwin focused on wealth, Emma focused on looks, they were married happily

## social cement

### families
- altruism makes sense for kin - wasps, ants
	- part of kin selection theory by Hamilton
	- some ants are sterile and only defend nests
- genes try to propagate *themselves* not individuals or groups
- $r$ - represents *degree of relatedness*
	- brother = 1/2
	- aunt = 1/4
	- cousin = 1/8
	- for some organisms, like slime mold, r=1
	- higher average r leads to more altruism
- children look after themselves first, then siblings
	- parents need to teach them to share
	- children are biologically inclined to listen to their parents when young
- biological evidence: wealthy people focus on boy children, poor on girls
	- measure in how many years after first child for next child
	- makes sense since male's reproductive potential is more affected by societal status
	- this same trend should show up for siblings (poor children are nicer to girl siblings)
- parents grieve most of children around adolescent age
	- this maps perfectly the reproductive potential of !Kung people

### darwin and the savages
- evolution: kin-selection -> reciprocal altruism (tit for tat) -> higher morals
- against group selection - even if it helped a group, it would start to decay within the group

### friends
- helping others is not zero-sum
	- lets you catch big game, spread information
- late 1970s Axelrod devises competition for prisoner's dillema programs
	- *TIT for TAT* programs wins - do what person did last
	- very simple for early ancestors to implement
	- designed for *individuals* not *groups*
- TIT for TAT doesn't work unless lots of people do it
	- kin selection gave it a boost
	- not sure about aunts, uncles, etc.
	- people try to maintain appearances of dignity
- once reciprocal altruism is entrenched, can have "good for the group" type genes

### darwin's conscience
- moral guidance is made to be guided by peers / parents
- lying can be useful
	- lying can be genetically made exciting to teach its usefulness
- modern society generally has more lying
	- this is to be expected as groups get larger and have more immigration and emigration

## social strife

### darwin's delay
- darwin studies barnacles for a while, probably because he was afraid to unveil his iconoclastic theory

### social status
- hierarchy exists almost everywhere
- even societies that have shared resources didn't share social status, females, etc.
- group-selection for hierarchy doesn't make sense - individual fitness in anarchy is fine
- 2 theories
	1. *pecking order*
		- if you leave hens, after some combat they settle into a pecking order
		- hen A pecks B with impunity, B pecks C, so on in order to settle disputes for resources
		- they all respect this
		- genes endowing a chicken with this selective fear should flourish
	2. *John Maynard Smith's* evolutionary *steady state*
		- hawk-dove analysis of birds
		- both strategies make sense in proportions
- males genetically tend to be most ambitious, seek social status
	- Sharifian emperor of Morocco credited with 888 children
- hierarchy comes after reciprocal altruism
	- status assistance could be main purpose of friendship
- *Machiavellianism* is "the employment of cunning and duplicity in statecraft or in general conduct"
- what we call cultural values are expedients to social success

### deception and self-deception
- it is more important to give the appearance of altruism than actually be altruistic
- some of our motives are hidden from us not incidentally but by design, so that we can credibly act as if they aren't what they are
- low *self-esteem* - way to reconcile people to subordinate status
- *galvanic skin response* (GSR) - rises if people hear their own voice
	- people somtimes can't consciously identify their own voice even when GSR can
	- they recognize it more when they have more confidence
- *glumness*
	1. self-esteem deflator
	2. negative reinforcement
	3. course changer
- *split-brain* patients make up reasons
	- one hemisphere gives them command walk
	- when asked why walking, they make something up
- NYT quote: "In a week's time, both sides have constructed deeply emotional stories explaining their roles, one-sided accounts that are offered with impassioned conviction, although in many respects they do not stand up, in either case, under careful scrutiny."

### darwin's triumph
- people were willing to profess incorrect opinions about the relative length of two lines if placed in a room with other people who professed them

## morals of the story

### darwinian (and freudian) cynicism
- Freud
	1. id - animal
	2. ego - interprets id to superego
	3. superego

### evolutionary ethics
- John Mill's *utilitarianism* is a good starting point
- morality best preserves non-zero sumness

### blaming the victim
- "genetic determinism" pops up in court cases (cases like insanity)
	- notion of free will is shrinking
- less retributive of justice - more emphasis on deterrence, improving utilitarianism
- rage of juries may wane as they come to believe that male philandering is "natural"

### darwin gets religion
- doctrines thus far likely have "harmony" with human nature
- we are designed to believe that next rung on ladder will bring bliss, but in reality it will evaporate shortly after we get there
- why religion
	1. power to religion makers
	2. mutual benefits for leaders and people
	3. we came to empathize with all people
	
### general tips
1. distinguish between behavior and mental organ governing it
2. remember that mental organ, not behavior, is what was actually designed by natural selection
3. these organs may no longer ba daptive
4. human mind is incredibly complex

# The Righteous Mind: Why Good People Are Divided by Politics and Religion

**jonathan haidt, 2012**

- questions
	- eating dead dog
	- ripping up american flag
	- sex with chicken
	- incest
- *parochial* - having a limited or narrow outlook or scope; of or relating to a church parish

## intuitions come first, strategic reasoning second
- elephant and rider metaphor

### where does morality come from
1. **nativist** - morality is innate
2. **empiricist** - morality is from childhood learning
3. **rationalist** - morality is self-constructed by children on the basis of their experience with harm
	- kids know harm is wrong because they hate to be harmed and learn its is wrong to harm others
	- came to reject this answer
- new study
	- moral domain *varies by culture*
		- rich, westerners tend to differentiate between social constructions and moral harms while others don't
		- westerners are *individualistic* - harm and fairness
		- other cultures are *sociocentric*
	- disgust and disrepect drive reasoning - moral reasoning is posthoc fabrication
- in fact, morality is probably some combination of 1 & 2

### intuitive dog and rational tail
1. Plato - reason (mind) is master of emotions
2. Hume - reason is the servant of passions
3. Jefferson - reason and sentiment are indpeendent co-rulers
- Haidt believes in 2
- Antonio Damasio writes Descartes' error where patients are missing ventromedial prefrontal cortex (vmPFC)
	- they couldn't have emotion
	- where difficult to reason without emotion - too many choices
- *intuitionism* - calls  reasoning *rider* and intution *elephant*
	- rider developed to help elephant
	- *social intuitionism* - other people can alter intuitions
	
### elephants rule
- brain can make snap judgements in 1/10 second
	- can predict 2/3 outcomes of senate / house elections based on attractiveness in this time
- intuitions come first, strategic reasoning second
- smells etc can influence our moral judgements

### vote for me (and here's why)
- conscious reasoning immediately justifies intuitive response
- self-esteem doesn't make evolutionary sense, since being in groups was what mattered
	- rather, self-esteem measure's one's fitness as a mate / group member
- experimental evidence for confirmation bias
- moral/political matters - we are often *groupish* rather than selfish


## there's more to morality than harm and fairness
- be suspicious of moral monomists

### beyond WEIRD mentality	
- WEIRD - western, educated, industrialized, rich, democratic - outliars, but often used
- Schweder's three ethics
	1. *autonomy* - individual rights
	2. *community* - group relationships
	3. *divinity* - purity
- there is more to morality than harm and fairness

### taste buds of the righteous mind / 7 - the moral foundations of politics
- *deontology* - rule-based ethics
- moral psychology should be empirical - how the mind works, not how it ought to work
- Moral Foundations Theory
	1. care
		- evolved to care for young
	2. fairness
		- punish cheaters
		- finding altruistic partner
	3. liberty
	4. loyalty
		- want people that are good team players
	5. authority
		- allows us to thrive in hierarchical settings
	6. sanctity
		- starts with omnivore's dillema
		- survive pathogens

### the conservative advantage
- Durkheim - basic unit is family, not individual
- liberals only really value first three moral foundations

## morality binds and blinds

### why are we so groupish
- group selection is controversial
- here are 3 exhibits defending it
	1. major transitions produce superorganisms
	2. shared intentionality generates moral matrices
		- chimpanzees have no shared intentionality
	3. genes and cultures coevolve
	4. evolution can be fast

### the hive switch
- two candidates for hive switch
	1. oxytocin genes
	2. mirror neurons
- hive switch doesn't seem to be for everyone, but rather just for one's group

### religion is a team sport	
- descriptive definitions - describe what people think are moral
- normative definitions - describe what is truly right
	- utilitarianism
	- deontology
- belief in supernatural - could be accidental as by-product of hypersensitive agency detection device
- religion can effectively surpress free-rider problem

### disagreeing more constructively
- people are predisposed to ideologies
- then there is serious confirmation bias
- liberals and conservatives are both necessary to balance each other out
- *Manichaeism* - polarization, believing one side only
- imagine world with no countries, religion -> would probably be chaos



# homo deus

## old problems: famine, war, plague

- famine, war, plague are less common now -- what will take their place?
  - if these + ecological equilibrium are solved, do we need more?
- plague
  - black death in 1330s - killed between 75-200 mil
  - smallpox plague in 1500s along with other diseases from Europe tothe Americas
  - spanish flu 1918 infectedabout 500 mil (1/3 of world population)
    - 50-100 mil died
  - covid19 (as of jan 2021)
    - ~1/2 mil dead
  - since 1980s, >30 mil AIDS death
  - "in the arms race between doctors and germs, doctors run faster"
- war kills many fewer people these days
  - mutually assured destruction
  - made things like information / knoweldge more important (e.g. can't loot tech)
  - Anton Chekhov famously said that a gun appearing in the first act of a play will inevitably be fired in the third ("chekhov's law")
    - nowadays in real world, may note be the case
  - terrorism generally works more by evoking outrage

## new goals: immortality, happiness and divinity

- immortality
  - ex. Ray Kurzweil at Google trying to "solve death"
  - in 20th century, life expectancy went from forty to seventy
  - no clear line separates healing from upgrading
- happiness
  - epicurus - happiness is goal of life
  - bentham/mill - happiness is pleasure - pain
  - suicide rates are ~25x higher in developed nations
  - 2 levels
    - psychological level: happiness depends on expectations rather than objective conditions
    - biological level: both our expectations and our happiness are determined by our biochemistry
  - new drugs are constantly being developed and societal standards around them shift
- divinity
  - biological engineering - rewriting genetic code
  - cyborg engineering - adding thinkgs like bionic hands, artificial eyes
  - engineering of non-organic beings - AIs
  - many of the the powers classical gods had are now possible through engineering
- Knowledge that does not change behaviour is useless. But knowledge that changes behaviour quickly loses its relevance
- the study of history aims above all to make us aware of possibilities we don’t normally consider. Historians study the past not in order to repeat it, but in order to be liberated from it.
  - ex. capitalism, feminism, civil rights
  - ex. lawns
  - ppl thought living without pharoahs was inconceivable



## PART I -- *Homo sapiens* Conquers the World

- emotions are algorithms imbued by genes
- religions sprung up w agriculture justifying animal cruelty
- the founding idea of humanist religions such as liberalism, communism and Nazism is that *Homo sapiens* has some unique and sacred essence that is the source of all meaning and authority in the universe
- no clear distinction between human and animals
  - evolution implies there is no eternal soul
- literal meaning of the word ‘individual’ is ‘something that cannot be divided’
- what happens in the mind that doesn't happen in the brain?
- possible that the sensations of consciousness / emotion are an unnecessary byproduct
- clever hans (math horse)
- *Homo sapiens* is the only species on earth capable of co-operating flexibly in large numbers
- individuals favor fairness (ex. ultimatum game) but societies tolerate inequality
- things can have subjective, objective, or intersubjective meaning (e.g. we agree money has value so it does)

## PART II *Homo Sapiens* Gives Meaning to the World

- Religion is *anything* that confers superhuman legitimacy on human social structures. It legitimises human norms and values by arguing that they reflect superhuman laws.
- 

# Predictably Irrational

**by Dan Ariely, 2008**

- **arbitrary coherence** - market prices themselves that influence consumers' willingness to pay. What this means is that demand is not, in fact, a completely separate force from supply.
- Choices are always relatives
	- Adding a comparable worse option makes the comparable option seem better
	- Supply and demand doesn’t always work
		- The price for black pearls was completely made up
		- People often stay anchored to the prices they first see
- Social norms compete with market norms
	- Fining parents who pick up their children late
- High price of ownership
	- Students who won tickets in a lottery would sell them for much more than buy them
	- Pepsi wins blind taste tests, coke wins shown ones
- [“dollar auction”](http://www.smbc-comics.com/index.php?id=3594)

# freud

- Id – set of instinctual trends
  - "contrary impulses exist side by side, without cancelling each other out. ... There is nothing in the id that could be compared with negation ... nothing in the id which corresponds to the idea of time."
- Ego – organized and realistic
- Super-ego – analyzes and moralizes – mediates between id and ego