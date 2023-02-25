---
layout: notes
title: fairness, sts
category: ai
typora-copy-images-to: ../assets
---

**Some notes on algorithm fairness and STS as applied to AI**

{:toc}

# fairness

## metrics

- good introductory [blog](https://towardsdatascience.com/a-tutorial-on-fairness-in-machine-learning-3ff8ba1040cb)
- causes of bias
  - skewed sample
  - tainted examples
  - selectively limited features
  - sample size disparity
  - proxies of sensitive attributes
- definitions
  - **unawareness** - don't show sensitive attributes
    - flaw: other attributes can still signal for it
  - group fairness
    - **demographic parity** - mean predictions for each group should be approximately equal
      - flaw: means might not be equal
    - **equalized odds** - predictions are independent of group given label
      - equality of opportunity: $p(\hat y=1|y=1)$ is same for both groups
    - **predictive rate parity** - Y is independent of group given prediction
  - **individual fairness** - similar individuals should be treated similarly
  - **counterfactual fairness** - replace attributes w/ flipped values
- fair algorithms
  - preprocessing - remove sensitive information
  - optimization at training time - add regularization
  - postprocessing - change thresholds to impose fairness



## [fairness in computer vision (tutorial)](https://sites.google.com/view/fatecv-tutorial/schedule)

### [harms of CV](https://www.youtube.com/watch?time_continue=546&v=0sBE5OyD7fk&feature=emb_logo)

- **timnit gebru** - fairness team at google
  - also emily denton
- startups
  - faceception startup - profile people based on their image
  - hirevue startup videos - facial recognition for judging interviews
  - clearview ai - search all faces
  - police using facial recognition - harms protestors
  - facial recognition rarely has good uses
  - contributes to mass surveillance
  - can be used to discriminate different ethnicities (e.g. Uighurs in china)
- gender shades work - models for gender classification were worse for black women
  - datasets were biased - PPB introduced to balance things somewhat
- gender recognition is harmful in the first place
  - collecting data without consent is also harmful
- [letter to amazon](https://www.theverge.com/2019/4/3/18291995/amazon-facial-recognition-technology-rekognition-police-ai-researchers-ban-flawed): stop selling facial analysis technology
- combating this technology
  - fashion for fooling facial recognition

### [data ethics](https://www.youtube.com/watch?v=v_XBJd1Fxqc&feature=emb_logo)

- different types of harms
  - sometimes you need to make sure there aren't disparate error rates across subgroups
  - sometimes the task just should not exist
  - sometimes the manner in which the tool is used is problematic because of who has the power
- technology amplifies our intent
- most people feel that data collection is the most important place to intervene
- people are denied housing based on data-driven discrimination
- collecting data
  - wild west - just collect everything
  - curatorial data - collect very specific data (this can help mitigate bias)
- datasets are value-laden, drive research agendas
- ex. celeba labels gender, attractiveness
- ex. captions use gendered language (e.g. beautiful)

### [where do we go?](https://www.youtube.com/watch?v=vpPpwa7W93I&feature=emb_logo)

- technology is not value-neutral -- it's political
- model types and metrics embed values
- science is not *neutral, objecive, perspectiveless*
- be aware of your own **positionality**
- concrete steps
  - ethics-informed model evaluations (e.g. disaggregegated evaluations, counterfactual testing)
  - recognize limitations of technical approaches
  - transparent dataset documentation
  - think about perspectives of marginalized groups

## facial rec. demographic benchmarking

- [gender shades](http://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf) (Buolamwini & gebru, 2018)
- [Face Recognition Vendor Test (FRVT) Part 3: Demographic Effects](https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.8280.pdf) (grother et al. 2019), NIST
  - facial rec types
    - 1: 1 == **verification**
    - 1: N == **identification**
  - data
  	- **domestic mugshots** collected in the United States
  	- **application photographs** from a global population of applicants for immigration benefits
  	- **visa photographs** submitted in support of visa applicants
  	- **border crossing photographs** of travelers entering the United States
  - a common practice is to use random pairs, but as the pairs are stratified to become more similar, the false match rate increases (Fig 3)
  - results
    - biggest errors seem to be in African Americans + East Asians
      - impact of errors - in verification, false positives can be security threat (while false negative is mostly just a nuisance)
    -  In domestic mugshots, false negatives are higher in Asian and American Indian individuals, with error rates above those in white and black face
      - possible confounder - aging between subsequent photos
    - better image quality reduces false negative rates and differentials
    - false positives to be between 2 and 5 times higher in women than men
    - one to many matching usually has same biases
      - a few systems have been able to remove bias in these false positives
    - did not analyze cause and effect
      - don't consider skin tone
- [Saving Face: Investigating the Ethical Concerns of Facial Recognition Auditing](https://dl.acm.org/doi/pdf/10.1145/3375627.3375820) (2020)

## legal perspectives

*“the master’s tools will never dismantle the master’s house.”*



**[ALGORITHMIC ACCOUNTABILITY: A LEGAL AND ECONOMIC FRAMEWORK](http://faculty.haas.berkeley.edu/morse/research/papers/AlgorithmicAccountability_BartlettMorseStantonWallace.pdf) (2020)**

- Title VII defines accountability under under U.S. antidiscrimination law
  - protected attributes: sex, religion, national origin, color
  - law is all about who has the burden of proof - 3 steps
    - **plaintiff** identifies practice that has observed statistical disparities on protected group
    - **defendant** demonstrates practice is (a) job-related (b) consistent with business necessity
    - **plaintiff** proposes alternative
- ex. dothard vs rawlingson: prison guards were selected based on weight/height rather than strength, so female applicants sued
  - legitimate target variable: strength
  - proxy variable: weight/height
  - supreme court ruled that best criterion is to assess strength, not use weight/height proxies
- ex. redlining - people in certain neighborhoods do not get access to credit
  - legitimate target: ability to pay back a loan
  - proxy variable: zip code (disproportionately affected minorities)
- 2 key questions
  - **legitimate target variable** - is unobservable target characteristic (e.g. strength) one that can justify hiring disparities?
    - disparate outcomes mus be justified by reference to a legitimate "business necessity" (e.g. for hiring, this would be a required job-related skill)
  - **biased proxy** - do proxy variables (e.g. weight/height) properly capture the legitimate target variable?
    - problematic "redundant encodings" - a proxy variable can be predictive of a legitimate target variable and membership in a protected group
- **input accountability test - captures these questions w/ basic statistics**
  - intuition: exclude input variables which are potentially problematic
    - in this context, easier to define fairness without tradeoffs
    - even in unbiased approach, still need things like subsidies to address systemic issues
  - the test
    - look at correlations between proxy and legitimate target, proxy and different groups - proxy should not systematically penalize members of a protected group
    - **regression form**
      - predict legitimate target from proxy: $Height_i = \alpha \cdot Strength_i + \epsilon_i$
      - measure if residuals are correlated with protected groups: $\epsilon_i \perp gender$
      - if they are correlated, exclude the feature
  - difficulties
    - target is often unobservable / has measurement err
    - have to define a threshold for testing residual correaltions (maybe 0.05 p-vaues)
    - there might exist nonlinear interactions
  - major issues
    - even if features are independently okay, when you combine them in a model the outputs can be problematic
- related approaches
  - some propose balancing the outcomes
    - one common problem here is that balancing err rates can force different groups to be different
  - some propose using bst predictive model alone
    - some have argued that a test for fairness is that there is no other algorithm that is as accurate and have less of an adverse impact (skanderson and ritter)
  - HUD's mere predictive test - only requires that prediction is good and that inputs are not subsitutes for a protected characteristic
- In a [2007 U.S. Supreme Court school-assignment case](https://www.edweek.org/policy-politics/districts-face-uncertainty-in-maintaining-racially-diverse-schools/2007/06) on whether race could be a factor in maintaining diversity in K-12 schools
    - Chief Justice John Roberts’ opinion famously concluded: “The way to stop discrimination on the basis of race is to stop discriminating on the basis of race.”
    - then-justice Ruth Bader Ginsburg said: “It’s very hard for me to see how you can have a racial objective but a nonracial means to get there.”


## misc papers

- [Large image datasets: A pyrrhic win for computer vision?](https://openreview.net/pdf?id=s-e2zaAlG3I) - bias in imagenet / tiny images
- http://positivelysemidefinite.com/2020/06/160k-students.html
- [How AI fails us](https://www.ethics.harvard.edu/files/center-for-ethics/files/howai_fails_us_2.pdf) (2021) - alternative to AEAI is not a singular, narrow focus on a specific goal (e.g. AGI), but support for a plurality of complementary and dispersed approaches to developing technology to support the plurality and plasticity of human goals that exist inside the boundaries of a human rights framework


# ethics

## some basics

- utilitarian - good to be maximized
  - utilitarian approach: which option produces the most good and does the least harm?
  - common good approach: which option best serves the community as a whole?
- deontological - adhering to "the right"
  - ex. principles like privacy, autonomy, conflicting 
  - rights approach: which option bbest respects the rights of all who have a stake?
  - justice approach: which option treats people equally or proportionately?
- consequentalist questions
  - who will be directly affected? who gets harmed?
- virtue ethics - lives by a code of honor
  - which option leas me to act as the sort of persion I want to be?

## moral trade

- [Moral Trade](https://www.fhi.ox.ac.uk/wp-content/uploads/moral-trade-1.pdf) (ord 2015) - **moral trade** = trade that is made possible by differences in the parties' moral views
- examples
  - one trading their eating meat for another donating more to a certain charity they both believe in
  - donating to/against political parties
  - donating to/against gun lobby
  - donating to/for pro-life lobby
  - paying non-profit employees
- benefits
  - can yield Pareto improvements = strict improvements where something gets better while other things remain atleast constant
- real-world examples
  - vote swapping (i.e. in congress)
  - vote swapping across states/regions (e.g. Nader Trader, VotePair) - ruled legal when money not involved
  - election campaign donation swapping - repledge.com (led by eric zolt) - was taken down due to issues w/ election financing
- issues
  - factual trust - how to ensure both sides carry through? (maybe financial penalties or audits could solve this)
  - counterfactual trust - would one party have given this up even if the other party hadn't?
- minor things
  - fits most naturally with moral framework of consequentalism
  - includes indexicals (e.g. prioritizing one's own family)
  - could have uneven pledges

# sts

- **social determinism** - theory that social interactions and constructs alone determine individual behavior
- **technological determinism** - theory that assumes that a society's technology determines the development of its social structure and cultural values
- [do artifacts have politics?](https://www.cc.gatech.edu/~beki/cs4001/Winner.pdf) (winner 2009)
    - **politics** - arrangements of power and authority in human associations as well as the activitites that take place within those arrangements
    - **technology** -  smaller or larger pieces or systems of hardware of a specific kind. 
    - examples
      - pushes back against social determinism - technologies have the ability to shift power
      - ex: nuclear power (consolidates power) vs solar power (democratizes power)
      - ex. tv enables mass advertising
      - ex. low bridges prevent buses
      - ex. automation removes the need for skilled labor
        - ex. tractors in grapes of wrath / tomato harvesters
      - ex. not making things handicap accessible
    - "scientific knowledge, technological invention, and corporate profit reinforce each other in deeply entrenched patterns that bear the unmistakable stamp of political and economic power"
      - pushback on use of things like pesticides, highways, nuclear reactors
    - technologies which are inherently political, regardless of use
      - ex. "If man, by dint of his knowledge and inventive genius has subdued the forces of nature, the latter avenge themselves upon him by subjecting him, insofar as he employs them, to a veritable despotism independent of all social organization."
      - attempts to justify strong authority on the basis of supposedly necessary conditions of technical practice have an ancient history. 
- [Disembodied Machine Learning: On the Illusion of Objectivity in NLP](https://openreview.net/pdf?id=fkAxTMzy3fs)
- [less work for mother](https://www.americanheritage.com/less-work-mother) (cowan 1987) - technologies that seem like they save time rarely do (although they increase "productivity")
- [The Concept of Function Creep](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3547903) - “Function creep denotes an imperceptibly transformative and therewith contestable change in a data-processing system’s proper activity.”



# effective altruism

- [effectivealtruism](https://www.effectivealtruism.org/articles/introduction-to-effective-altruism/)
    - promising causes
      - great in *scale* (it affects many lives, by a great amount)
      - highly *neglected* (few other people are working on addressing the problem), and
      - highly *solvable* or *tractable* (additional resources will do a great deal to address it).
    - 3 big areas
      - fighting extreme poverty (e.g. malaria)
      - animal suffering
      - improving the long-term future
- [rethink-priorities jobs](https://rethinkpriorities.freshteam.com/jobs/iX8GfQ1eBLDq/researcher-multiple-positions-remote)
- [open philanthropy](https://www.openphilanthropy.org/blog/modeling-human-trajectory)
- careers can have greater impacts than altruism ([80k hours](https://80000hours.org/key-ideas/))
    - https://80000hours.org/career-guide/
    - [80k hours AI careers](https://80000hours.org/problem-profiles/positively-shaping-artificial-intelligence/)
- [givewell cost-effective calculations](https://www.givewell.org/how-we-work/our-criteria/cost-effectiveness)
    - charities often exaggerate their "cost to save a life"
- [The case for reducing existential risks - 80,000 Hours](https://80000hours.org/articles/existential-risks/)
    - life-ending asteroid: 1 in 5000 per century
        - supervolcanoes

    - nuclear weapons
    - biochemical weapons
    - runaway climate change
    - unknown unknowns


# tech for "good"

- [Essays: Cybersecurity for the Public Interest - Schneier on Security](https://www.schneier.com/essays/archives/2019/02/public-interest_tech.html)

- [Machine Learning that Matters](https://arxiv.org/abs/1206.4656) (wagstaff, 2021)
  - doesn't make sense to compare the same metrics across different dsets (e.g. AUC over entire domain doesn't matter)
  - classification / regr. are overemphasized ([langley, 2011](https://link.springer.com/content/pdf/10.1007/s10994-011-5242-y.pdf))

# concrete ai harms

- [awful-ai](https://github.com/daviddao/awful-ai) - list of scary usages of ai

Technologies, especially world-shaping technologies like CNNs, are never objective. Their existence and adoption change the world in terms of 

- consolidation of power (e.g. facial-rec used to target Uighurs, increased rationale for amassing user data)

- a shift toward the quantitative (which can lead to the the type of click-bait extremization we see online)

- automation (low-level layoffs, which also help consolidate power to tech giants)

- energy usage (the exorbitant footprint of models like GPT-3)

- access to media (deepfakes, etc.)

- a lot more

- pandemic

  - I hope the pandemic, which has boosted the desire for tracking, does not result in a long-term arc towards more serveillance
  - from [here](https://www.theatlantic.com/magazine/archive/2020/09/china-ai-surveillance/614197/): City Brain would be especially useful in a pandemic. (One of Alibaba’s sister companies created the app that color-coded citizens’ disease risk, while silently sending their health and travel data to police.) As Beijing’s outbreak spread, some malls and restaurants in the city began scanning potential customers’ phones, pulling data from mobile carriers to see whether they’d recently traveled. Mobile carriers also sent municipal governments lists of people who had come to their city from Wuhan, where the coronavirus was first detected. And Chinese AI companies began making networked facial-recognition helmets for police, with built-in infrared fever detectors, capable of sending data to the government. City Brain could automate these processes, or integrate its data streams.
  - "The pandemic may even make people value privacy less, as one early poll in the U.S. suggests"

  