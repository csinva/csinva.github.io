---
layout: notes
title: fairness
category: ai
typora-copy-images-to: ./assets/nlp
---

{:toc}

Some notes on algorithm fairness and STS.


# fairness metrics

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



# [fairness in cv (tutorial)](https://sites.google.com/view/fatecv-tutorial/schedule)

## [Computer vision in practice: who is benefiting and who is being harmed?](https://www.youtube.com/watch?time_continue=546&v=0sBE5OyD7fk&feature=emb_logo)

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

## [data ethics](https://www.youtube.com/watch?v=v_XBJd1Fxqc&feature=emb_logo)

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

## [where do we go?](https://www.youtube.com/watch?v=vpPpwa7W93I&feature=emb_logo)

- technology is not value-neutral -- it's political
- model types and metrics embed values
- science is not *neutral, objecive, perspectiveless*
- be aware of your own **positionality**
- concrete steps
  - ethics-informed model evaluations (e.g. disaggregegated evaluations, counterfactual testing)
  - recognize limitations of technical approaches
  - transparent dataset documentation
  - think about perspectives of marginalized groups

## misc papers

- [Large image datasets: A pyrrhic win for computer vision?](https://openreview.net/pdf?id=s-e2zaAlG3I) - bias in imagenet / tiny images
- http://positivelysemidefinite.com/2020/06/160k-students.html

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



# facial rec. demographic benchmarking

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
  - 



# legal perspectives

- **“the master’s tools will never dismantle the master’s house.”**

## [ALGORITHMIC ACCOUNTABILITY: A LEGAL AND ECONOMIC FRAMEWORK](http://faculty.haas.berkeley.edu/morse/research/papers/AlgorithmicAccountability_BartlettMorseStantonWallace.pdf) (2020)

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

### **input accountability test - captures these questions w/ basic statistics**

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

### **related approaches**

- some propose balancing the outcomes
  - one common problem here is that balancing err rates can force different groups to be different
- some propose using bst predictive model alone
  - some have argued that a test for fairness is that there is no other algorithm that is as accurate and have less of an adverse impact (skanderson and ritter)
- HUD's mere predictive test - only requires that prediction is good and that inputs are not subsitutes for a protected characteristic