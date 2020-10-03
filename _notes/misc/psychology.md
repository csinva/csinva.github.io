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

- Aristotle’s 4 "causes" or modes of explanation

  | cause         | description                                           | example                                          |
  | :------------ | ----------------------------------------------------- | ------------------------------------------------ |
  | **efficient** | proximal  mechanisms of change                        | a carpenter is an efficient cause of a bookshelf |
  | **final**     | the end, function or goal                             | holding books is a final cause of a bookshelf    |
  | **formal**    | the form or properties that make something what it is | having shelves is a formal cause of a bookshelf  |
  | **material**  | the substance of which something is constituted       | wood is a material cause of a bookshelf          |

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

## Causal Explanation (lombrozo & vailyeva 2017)

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
- engaging in explanation influence causal model learning



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



# freud

- Id – set of instinctual trends
  - "contrary impulses exist side by side, without cancelling each other out. ... There is nothing in the id that could be compared with negation ... nothing in the id which corresponds to the idea of time."
- Ego – organized and realistic
- Super-ego – analyzes and moralizes – mediates between id and ego