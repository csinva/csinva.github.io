---
layout: notes
section-type: notes
title: Artificial Intelligence
category: ai
---

* TOC
{:toc}

[toc]

# symbol search
- computer science - empirical inquiry

## symbols and physical symbol systems
- intelligence requires the ability to store and manipulate symbols
- laws of qualitative structure
	- cell doctrine in biology
	- plate tectonics in geology
	- germ theory of disease
	- doctrine of atomism
- "physical"
	1. obey laws of physics
	2. not restricted to human systems
	- *designation* - then given the expression, the system can affect the object
	- *interpretation* - expression designates a process
- *physical symbol system hypothesis* - a physical symbol system has the necessary and sufficient means for general intelligent action
    - from { cite newell1980physical }
	- identify a task domain calling for intelligence; then construct a program for a digital computer that can handle tasks in that domain
- no boundaries have come up yet
- wanted general problem solver - leads to generalized schemes of representation
- goes along with information processing psychology
	- observe human actions requiring intelligence
	- program systems to model human actions

## heuristic searching
- symbol systems solve problems with *heuristic search*
- *Heuristic Search Hypothesis* - solutions are represented as symbol structures. A physical symbol system exercises its intelligence in problem solving by search--that is, by generating and progressively modifying symbol structures until it produces a solution structure
    - from { cite newell1976computer }
- there are practical limitations on how fast computers can search
- To state a problem is to designate
    1. a test for a class of symbol structures (solutions of the problem)
    2. a generator of symbol structures (potential solutions). 
- To solve a problem is to generate a structure, using (2), that satisfies the test of (1).
- searching is generally in a tree-form

# intro
- AI - field of study which studies the goal of creating intelligence
	- *intelligent agent* - system that perceives its environment and takes actions that maximize its chances of success
- expert task examples - medical diagnosis, equipment repair, computer configuration, financial planning
1. formal systems - use axioms and formal logic
2. *ontologies* - structuring knowledge in graph form
3. statistical methods
- *turing test* - is human mind deterministic { turing1950computing }
- *chinese room argument* - rebuts turing test { cite searle1980minds }
- *china brain* - what if different people hit buttons to fire individual neurons

# knowledge representation
- *physical symbol system hypothesis* - a physical symbol system has the necessary and sufficient means for general intelligent action
	- computers and minds are both *physical symbol systems*
	- *symbol* - meaningful pattern that can be manipulated
	- symbol system - creates, modifies, destroys symbols
- want to represent
	1. *meta-knowledge* - knowledge about what we know
	2. *objects* - facts
	3. *performance* - knowledge about how to do things
	4. *events* - actions
- two levels
	1. knowledge level - where facts are described
	2. symbol level - lower
- properties
	1. representational adequacy - ability to represent
	2. inferential adequacy
	3. inferential efficiency
	4. acquisitional efficiency - acquire new information
- two views of knowledge
	1. logic
		- a *logic* is a language with concrete rules
		- *syntax* - rules for constructing legal logic
		- *semantics* - how we interpret / read
			- assigns a meaning
		- multi-valued logic - not just booleans
		- higher-order logic - functions / predicates are also objects
		- multi-valued logics - more than 2 truth values
			- fuzzy logic - uses probabilities rather than booleans
		- match-resolve-act cycle
	2. associationist
		- knowledge based on observation
		- semantic networks - objects and relationships between them			- like is a, can, has
			- *graphical representation*
			- equivalent to logical statements
			- ex. nlp - conceptual dependency theory - sentences with same meaning have same graphs
			- *frame representations* - semantic networks where nodes have structure
				- ex. each frame has age, height, weight, ...
			- when agent faces *new situation* - slots can be filled in, may trigger actions / retrieval of other frames
			- inheritance of properties between frames
			- frames can contain relationships and procedures to carry out after various slots filled

# expert systems
- *expert system* - program that contains some of the subject-specific knowledge of one or more human experts.
- problems
	1. planning
	2. monitoring
	3. instruction
	4. control
- need lots of knowledge to be intelligent
- *rule-based architecture* - condition-action rules & database of facts
- acquire new facts
	- from human operator
	- interacting with environment directly
- forward chaining
	- until special HALT symbol in DB, keep following logical rule, add result to DB
- conflict resolution - which rule to apply when many choices available
- *pattern matching* - logic in the if statements
- *backward chaining* - check if something is true
	- check database
	- check if on the right side of any facts
- *CLIPS* - expert system shell
	- define rules and functions...
- *explanation subsystem* - provide explanation of reasoning that led to conclusion
- people
	1. *knowledge engineer* - computer scientist who designs / implements ai
	2. *domain expert* - has domain knowledge
1. user interface
2. knowledge engineering - art of designing and building expert systems
	- determine characteristics of problem
	- *automatic knowledge-acquisition* - set of techniques for gaining new knowledge
		- ex. parse Wikipedia
		- crowdsourcing
- creating an expert system can be very hard
	- only useful when expert isn't available, problem uses symbolic reasoning, problem is well-structured
- *MYCIN* - one of first successful expert systems { cite shortliffe2012computer }
	- Stanford in 1970s
	- used backward chaining but would ask patient questions - sometimes too many questions
- advantages
	- can explain reasoning
	- can free up human experts to deal with rare problems

# decisions

## game trees -- R&N 5.2-5.5
- *minimax algorithm*
	- *ply* - half a move in a tree
	- for multiplayer, the backed-up value of a node n is the vector of the successor state with the highest value for the player choosing at n
	- time complexity - $O(b^m)$
	- space complexity - O(bm) or even O(m)
- *alpha-beta* pruning cut in half the exponential depth
	- once we have found out enough about n, we can prune it
	- depends on move-ordering
		- might want to explore best moves = *killer moves* first
	- *transposition table* can hash different movesets that are just transpositions of each other
- imperfect real-time decisions
	- can evaluate nodes with a heuristic and cutoff before reaching goal
	- heuristic uses features
	- want *quiescent search* - consider if something dramatic will happen in the next ply
	- *horizon effect* - a position is bad but isn't apparent for a few moves
	- *singular extension* - allow searching for certain specific moves that are always good at deeper depths
	- *forward pruning* - ignore some branches
		- *beam search* - consider only n best moves
		- PROBCUT prunes some more
- search vs lookup
	- often just use lookup in the beginning
	- program can solve and just lookup endgames
- stochastic games
	- include *chance nodes*
	- change minimax to expectiminimax
	- $O(b^m numRolls^m)$
	- cutoff evaluation function is sensitive to scaling - evaluation function must be a postive linear transformation of the probability of winning from a position
	- can do alpha-beta pruning analog if we assume evaluation function is bounded in some range
	- alternatively, could simulate games with *Monte Carlo simulation*
	

## utilities / decision theory -- R&N 16.1-16.3
- $P(RESULT(a)=s'|a,e)$
	- s - state, observations e, action a
- utility function U(s)
- rational agent should choose action  with *maximum expected utility*
	- expected utility $EU(a|e) = \sum_{s'} P(RESULT(a)=s'|a,e) U(s')$
- notation
	- A>B - agent prefers A over B
	- A~B - agenet is indifferent between A and B
- preference relation has 6 *axioms of utility theory*
	1. *orderability* - A>B, A~B, or A<B
	2. *transitivity*
	3. *continuity*
	4. *substitutability* - can do algebra with preference eqns
	5. *monotonicity* - if A>B then must prefer higher probability of A than B
	6. *decomposability* - 2 consecutive lotteries can be compressed into single equivalent lottery
- these axioms yield a utility function
	- isn't unique (ex. affine transformation yields new utility function)
	- sometimes ranking, not numbers needed - *value function* = *ordinal utility function*
	- agent might not be explicitly maximizing the utility function

### utility functions
- *preference elicitation* - finds utility function
	- normalized utility to have min and max value
	- assess utility of s by asking agent to choose between s and (p:min, (1-p):max)
	- *micromort* - one in a million chance of death
	- *QALY* - quality-adjusted life year
- money
	- agenets exhibits *monotonic preference* for more money
	- gambling has expected monetary value = EMV
	- when utility of money is sublinear - *risk averse*
		- value agent will accept in lieu of lottery = *certainty equivalent*
		- EMV - certainty equivalent = *insurance premium*
	- when supralinear - *risk-seeking* or linear - *risk-neutral*
- *optimizer's curse* - tendency for E[utility] to be too high
- *descriptive theory* - how actual agents work
	- decision theory - *normative theory*
	- *certainty effect* - people are drawn to things that are certain
	- *ambiguity aversion*
	- *framing effect* - wording can influence people's judgements
		- *evolutionary psychology*
	- *anchoring effect* - buy middle-tier wine because expensive is there
		
## decision theory / VPI -- R&N 16.5 & 16.6
- *decision network*
	1. *chance nodes* - represent RVS (like BN)
	2. *decision nodes* - points where decision maker has a choice of actions
	3. *utility nodes* - represent agent's utility function
- can ignore chance nodes
	- then *action-utility function* = *Q-function* maps directly from actions to utility
- evaluation
	1. set evidence
	2. for each possible value of decision node
		- set decision node to that value
		- calculate probabilities of parents of utility node
		- calculate resulting utility
	3. return action with highest utility

### the value of information
- *information value theory* - enables agent to choose what info to acquire
	- observations only effect agents belief state
	- value of info = expected value between best actions before and after info is obtained
- *value of perfect information VPI* - assume we can obtain exact evidence on some variable $e_j$
	- $VPI_e(E_j) = \left(\sum_k P(E_j = e_{jk}|e) \: EU(\alpha_{ejk} | e, E_j = e_{jk})\right) - EU(\alpha|e)$
	- info is more valuable when it is likely to cause a change of plan
	- info is more valuable when the new plan will be much better than the old plan
	- VPI not linearly additive, but is order-independent
- information-gathering agent
- *myopic* - greedily obtain evidence which yields highest VPI until some threshold
- *conditional plan* - considers more things
	
## mdps and rl -- R&N 17.1-17.4, 21.1-21.6
- *fully observable* - agent knows its state
- *markov decision process*
	- set of states
	- set of actions
	- transition model $P(s'|s,a)$
	- reward function R(s)
	- solution is policy $\pi^* (s)$ - what action to do in state s
		- optimal policy yields highest expected utlity
- optimizing MDP - *multiattribute utility theory*
	- could sum rewards, but results are infinite
	- instead define objective function (maps infinite sequences of rewards to single real numbers)
		- ex. set a *finite horizon* and sum rewards
			- optimal action in a given state could change over time = *nonstationary*
		- ex. discounting to prefer earlier rewards (most common)
			- could discount reward n steps away by $\gamma^n$, 0<r<1
		- ex. average reward rate per time step
		- ex. agent is guaranteed to get to terminal state eventually - *proper policy*
- expected utility executing $\pi$: $U^\pi (s) = E[\sum_t \gamma^t R(S_t)]$
	- when we use discounted utilities, $\pi$ is independent of starting state
	- $\pi^*(s) = \underset{\pi}{argmax \: U^\pi (s)} = \underset{a}{argmax} \sum_{s'} P(s'|s,a) U'(s)$ - utility of state is immediate reward for that state plus the expected discounted utility of the next state, assuming agent chooses optimal action
	
### value iteration
- *value iteration* - calculates utility of each state and uses utilities to find optimal policy
	- *bellman eqn* - $U(s) = R(s) + \gamma \: \underset{a}{max} \sum_{s'} P(s'|s, a) U(s')$
	- start with arbitrary utilities
	- recalculate several times with *Bellman update* to approximate solns to bellman eqn
		= $U_{i+1}(s) = R(s) + \gamma \: \underset{a}{max} \sum_{s'} P(s'|s, a) U_i(s')$
- value iteration eventually converges
	- *contraction* - function of one variable that when applied to 2 different inputs in turn produces 2 output values that are closer together than the original inputs
		- contraction only has 1 fixed point
	- Bellman update is a contraction on the space of utility vectors and therefore converges
	- error is reduced by factor of $\gamma$ each iteration
	- also, terminating condition -  if $||U_{i+1}-U_i|| < \epsilon (1-\gamma) / \gamma$ then $||U_{i+1}-U||<\epsilon$
	- what actually matters is *policy loss* $||U^{\pi_i}-U||$ - the most the agent can lose by executing $\pi_i$ instead of the optimal policy $\pi^*$
		- if $||U_i -U|| < \epsilon$ then $||U^{\pi_i} - U|| < 2\epsilon \gamma / (1-\gamma)$
		
### policy iteration
- another way to find optimal policies
	1. *policy evaluation* - given a policy $\pi_i$, calculate $U_i=U^{\pi_i}$, the utility of each state if $\pi_i$ were to be executed
		- like value iteration, but with a set policy so there's no max
		- can solve exactly for small spaces, or approximate
	2. *policy improvement* - calculate a new MEU policy $\pi_{i+1}$ using one-step look-ahead based on $U_i$
		- same as above, just $\pi^*(s) = \underset{\pi}{argmax \: U^\pi (s)} = \underset{a}{argmax} \sum_{s'} P(s'|s,a) U'(s)$
- *asynchronous policy iteration* - don't have to update all states at once

### partially observable markov decision processes (POMDP)
- agent is not sure what state it's in
- same elements but add *sensor model* P(e|s)
- have prob. distr b(s) for belief states
	- updates like the HMM
	- $b'(s') = \alpha P(e|s') \sum_s P(s'|s, a) b(s)$
	- changes based on observations
- optimal action depends only on the agent's current belief state - use belief states as the states of an MDP and solve as before
	- changes because state space is now continuous
- value iteration
	1. expected utility of executing p in belif state is just $b \cdot \alpha_p$ - dot product
	2. $U(b) = U^{\pi^*}(b)=\underset{p}{max} \: b \cdot \alpha_p$
	- belief space is continuous [0,1] so we represent it as piecewise linear, and store these discrete lines in memory
		- do this by iterating and keeping any values that are optimal at some point
			- remove *dominated plans*
	- generally this is far too inefficient
- *dynamic decision network* - online agent ![](online_pomdp.jpg) 
	- still don't really understand this

## reinforcement learning
- *reinforcement learning* - use observed rewards to learn optimal policy for the environment
- 3 agent designs
	1. *utility-based agent* - learns utility function on states
		- requires model of the environment
	2. *Q-learning agent*
		- learns *action-utility function* = *Q-function* maps directly from actions to utility
	3. *reflex agent* - learns policy that maps directly from states to actions

### passive reinforcement learning
- given policy $\pi$, learn $U^\pi (s)$
- like policy evaluation, but transition model / reward function are unknown
- *direct utility estimation* - run a bunch of trials to sample utility = expected total reward from each state
- *adaptive dynamic programming* (ADP) - learn transition model and rewards, and then plug into Bellman eqn
	- *prioritized sweeping* - prefers to make adjustements to states whose likely succesors have just undergone a large adjustment in their own utility estimates
- two ways to add prior
	1. *Bayesian reinforcement learning* - assume a prior P(h) on the transition model
		- use prior to calculate $P(h|e)$
		- let $u_h^\pi$ be expected utility avareaged over all possible start states, obtained by executing policy $\pi$ in model h
		- $\pi^* = \underset{\pi}{argmax} \sum_h P(h|e) u_h^\pi$
	2. give best outcome in the worst case over H (from *robust control theory*)
		- $\pi^* = \underset{\pi}{argmax} \underset{h}{min} u_h^\pi$
- *temporal-difference learning* - adjust utility estimates towards the ideal equilibrium that holds locally when the utility estimates are correct
	- $U^\pi = U^\pi (s) + \alpha (R(s) + \gamma U^\pi (s') - U^\pi (s))$
	- like a crude approximation of ADP
	
### active reinforcement learning
- *explore* states to find their utilities and *exploit* model to get highest reward
- *bandit* problems - determining exploration policy
	- should be *GLIE* - greedy in the limit of infinite exploration - visits all states infinitely, but eventually become greedy
		- ex. choose random action 1/t of the time
		- better ex. give optomistic prior utility to unexplored states
			- uses *exploration function* f(u,numTimesVisited) in utility update rule
	- *n-armed bandit* - pulling n levelers on a slot machine, each with different distr.
		- *Gittins index* - function of number of pulls / payoff
			
### learning action-utility function
- U(s) = $\underset{a}{max} Q(s,a)$
	- does require $P(s'|s,a)$ if we use ADP
	- doesn't require knowing $P(s'|s,a)$ if we use TD: $Q(s,a) = Q(s,a) + \alpha (R(s) + \gamma \underset{a'}{max} Q(s', a') - Q(s,a))$
- *SARSA* is related: $Q(s,a) = Q(s,a) + \alpha (R(s) + \gamma Q(s', a') - Q(s,a))$
	- here, a' is action actually taken
	- SARSA is *on-policy* while Q-learning is *off-policy*
	
### generalization
- approximate Q-function
	- ex. linear function of parameters
		- can learn params online with *delta rule* = *wildrow-hoff rule*: $\theta_i = \theta - \alpha \: \frac{\partial Loss}{\partial \theta_i}$
		
### policy search
- keep twiddling the policy as long as it improves, then stop
	- store one Q-function (parameterized by $\theta$) for each action
	- $\pi(s) = \underset{a}{max} \hat{Q}_\theta (s,a)$
		- this is discontinunous, instead often use *stochastic policy* representation (ex. softmax for $\pi_theta (s,a)$)
- learns $\theta$ that results in good performance
	- Q-learning learns actual Q* function - coulde be different (scaling factor etc.)
- to find $\pi$ maximize *policy value* $p(\theta)$
	- could do this with gradient ascient / empirical gradient hill climbing
- when environment/policy is stochastic, more difficult
	1. could sample mutiple times to compute gradient
	2. REINFORCE algorithm - could approximate gradient at $\theta$ by just sampling at $\theta$: $\nabla_\theta p(\theta) \approx \frac{1}{N} \sum_{j=1}^N \frac{(\nabla_\theta \pi_\theta (s,a_j)) R_j (s)}{\pi_\theta (s,a_j)}$
	3. PEGASUS - *correlated sampling* - ex. 2 blackjack programs would both be dealt same hands
	
### applications
- game playing
- robot control

# logic and planning
- *knowledge-based agents* - intelligence is based on *reasoning* that operates on internal *representations of knowledge*

## logical agents - 7.1-7.7 (omitting 7.5.2)
- 3 steps'- given a percept, the agent 
	1. adds the percept to its knowledge base
	2. asks the knowledge base for the best action
	3. tells the knowledge base that it has taken that action
- *declarative* approach - tell sentences until agent knows how to opearte
- *procedural* approach - encodes desired behaviors as program code
- ex. Wumpus World
- logical *entailment* between senteces
	- B follows logically from A (A implies B)
	- $A \vDash B$
- *model checking* - try everything to see if A $\implies$ B
	- this is *sound*=*truth-preserving*
	- *complete* - can derive any sentence that is entailed
- *grounding* - connection between logic and real environment (usually sensors)
- inference
	- TT-ENTAILS - recursively enumerate all sentences - check if a query is in the table
- theorem properties
	- *satisfiable* - true under some model
	- *validity* - *tautalogy* - true under all models
	- *monotonicity* - set of impliciations can only increase as info is added to the knowledge base
		- if $KB \implies A$ then $KB \land B \implies A$

## theorem proving
- *resolution rule* - resolves different rules with each other - leads to complete inference procedure
- *CNF* - *conjunctive normal form* - conjunction of clauses 
	- anything can be expressed as this
- *skip this* - resolution algorithm: check if $KB \implies A$ so check if $KB \land -A$
	- keep adding clauses until 
		1. nothing can be added
		2. get empty clause so $KB \implies A$
	- *ground resolution thm* - if a set of clauses is unsatisfiable, then the resolution closure of those clauses contains the empty clause
		- *resolution closure* - set of all clauses derivable by repeated application of resolution rule
- restricted knowledge bases
	- *horn clause* - at most one positive
		- *definite clause* - disjunction of literals with exactly one positive
		- *goal clause* - no positive
		- benefits
			- easy to understand
			- forward-chaining / backward-chaining are applicable
			- deciding entailment is linear in size(KB)
	- forward/backward chaining
		- checks if q is entailed by KB of definite clauses
			- keep adding until query is added or nothing else can be added
		- backward chaining works backwards from the query
- checking satisfiability
	- complete backtracking
		- *davis-putnam* algorithm = *DPLL* - like TT-entails with 3 improvements
			1. early termination
			2. pure symbol heuristic - *pure symbol* appears with same sign in all clauses
			3. unit clause heuristic - clause with just on eliteral or one literal not already assigned false
		- other improvements (similar to search)
			1. component analysis
			2. variable and value ordering
			3. intelligent backtracking
			4. random restarts
			5. clever indexing
	- local search
		- evaluation function can just count number of unsatisfied clauses (MIN-CONFLICTS algorithm for CSPs)
		- WALKSAT - randomly chooses between flipping based on MIN-CONFLICTS and randomly
			- runs forever if no soln
	- *underconstrained* problems are easy to find solns too
	- *satisfiability threshold conjecture* - for random clauses, probability of satisfiability goes to 0 or 1 based on ratio of clauses to symbols
		- hardest problems are at the threshold
		- state variables that change over time also called *fluents*
			- can index these by time
	- *effect axioms* - specify the effect of an action at the next time step
	- *frame axioms* - assert that all propositions remain the same
	- *succesor-state axiom*: $F^{t+1} \iff  ActionCausesF^t \lor (F^t \land -ActionCausesNotF^t )$
- keeping track of *belief state*
	- can just use 1-CNF
		- 1-CNF includes all states that are in fact possible given the full percept history
		- *conservative approximation*
- *SATPLAN* - how to make plans for future actions that solve the goal by propositional inference
	- must add *precondition axioms* - states that action occurrence requires preconditions to be satisfied
	- *action exclusion axioms* - one action at a time
	
## first-order logic - 8.1-8.3.3
- declarative language - semantics based on a truth relation between sentences and possible worlds
	- has *compositionality* - meaning decomposes
- *Sapir-Whorf hypothesis* - understanding of the world is influenced by the language we speak
- 3 elements
	1. objects
	2. relations
		- functions - only one value for given input
- first-order logic assumes more about the world than propositional logic
	- *epistemological commitments* - the possible states of knowledge that a logic allows with respect to each fact
	- *higher-order logic* - views relations and functions as objects in themselves
- first-order consists of symbols
	1. *constant symbols* - stand for objects
	2. *predicate symbols* - stand for relations
	3. *function symbols* - stand for functions
	- *arity* - fixes number of args
	- *term* - logical expresision tha refers to an object'
	- *atomic sentence* - formed from a predicate symbol optionally followed by a parenthesized list of terms
		- true if relation holds among objects referred to by the args
	- $\forall, \exists$, etc.
	- *interpretation* - specifies exactly which objects, relations and functions are referred to by the symbols

## inference in first-order logic - 9.1-9.4
- *propositionalization* - can convert first-order logic to propositional logic and do propositional inference
	- *universal instantiation* - we can infer any sentence obtained by substituting a ground term for the variable
		- replace "forall x" with a specific x
	- *existential instantiation* - variable is replaced by a new constant symbol
		- replace "there exists x" with a specific x
		- *Skolem constant* - new name of constant
	- only need finite subset of propositionalized KB - can stop nested functions at some depth
		- *semidecidable* - algorithms exist that say yes to every entailed sentence, but no algorithm exists that also says no to every nonentailed sentence
- *generalized modus ponens*
- *unification* - finding substitutions that make different logical expressions look identical
	- UNIFY(Knows(John,x), Knows(x,Elizabeth)) = fail .
		- use different x's - *standardizing apart*
		- want most general uniier
	- need *occur check* - S(x) can't unify with S(S(x))
- storage and retrieval
	- STORE(s) - stores a sentence s into the KB
	- FETCH(q) - returns all unifiers such that the query q unifies with some sentence in the KB
		- only try to unfity reasonable facts using *indexing*
		- query such as Employs(IBM, Richard)
		- all possible unifying queries form *subsumption lattice*
- forward chaining
	- *first-order definite clauses* - disjunctions of literals of which exactly one is positive (could also be implication whose consequent is a single positive literal)
	- *Datalog* - language restricted to first-order definite clauses with no function symbols
	- simple forward-chaining: FOL-FC-ASK
		1. *pattern matching* is expensive
		2. rechecks every rule
		3. generates irrelevant facts
	- efficient forward chaining (solns to above problems)
		1. *conjuct odering* problem - find an ordering to solve the conjuncts of the rule premise so the total cost is minimized
			- requires heuristics (ex. *minimum-remaining-values*)
		2. incremental forward chaining - ignore redundant rules
			- every new fact inferred on iteration t must be derived from at least one new fact inferred on iteration t-1
			- *rete* algorithm was first to do this
		3. irrelevant facts can be ignored by backward chaining
			- could also use *deductive database* to keep track of relevant variables
- backward-chaining
	- simple backward-chaining: FOL-BC-ASK
		- is a *generator* - returns multiple times, each giving one possible result
	- logic programming: algorithm = logic + control
		- ex. *prolog*
		- a lot more here
		- can have parallelism
		- redudant inference / infinite loops because of repeated states and infinite paths
		- can use *memoization* (similar to the dynamic programming that forward-chaining does)
		- generally easier than converting it into FOLD
	- *constraint logic programming* - allows variables to be constrained rather than *bound*
		- allows for things with infinite solns
		- can use *metarules* to determine which conjuncts are tried first
		
## classical planning 10.1-10.2
- *planning* - devising a plan of action to achieve one's goals
- *Planning Domain Definition Language* (PDDL) - uses *factored representation* of world
- *closed-world* assumption - fluents that aren't present are false
- set of ground (variable-free) actions can be represented by a single *action schema*
	- like a method

### algorithms for planning as state-space search



## knowledge representation 12.1 - 12.3
- *ontological engineering* - representing objects and their relationships
	- upper ontology - tree more general at the top more specific at bottom
- must represent *categories*
	- subcategories make a *taxonomy*
- can also define functions
	- *mass noun* - function that includes only *intrinsic* properties
	- *count noun* - function that includes any *extrinsic* properties
- *physical symbol system hypothesis* - a physical symbol system has the necessary and sufficient means for general intelligent action
	- computers and minds are both *physical symbol systems*
	- *symbol* - meaningful pattern that can be manipulated
	- symbol system - creates, modifies, destroys symbols
- want to represent
	1. *meta-knowledge* - knowledge about what we know
	2. *objects* - facts
	3. *performance* - knowledge about how to do things
	4. *events* - actions
- two levels
	1. knowledge level - where facts are described
	2. symbol level - lower
- properties
	1. representational adequacy - ability to represent
	2. inferential adequacy
	3. inferential efficiency
	4. acquisitional efficiency - acquire new information
- two views of knowledge
	1. logic
		- a *logic* is a language with concrete rules
		- *syntax* - rules for constructing legal logic
		- *semantics* - how we interpret / read
			- assigns a meaning
		- multi-valued logic - not just booleans
		- higher-order logic - functions / predicates are also objects
		- multi-valued logics - more than 2 truth values
			- fuzzy logic - uses probabilities rather than booleans
		- match-resolve-act cycle
	2. associationist
		- knowledge based on observation
		- semantic networks - objects and relationships between them			- like is a, can, has
			- *graphical representation*
			- equivalent to logical statements
			- ex. nlp - conceptual dependency theory - sentences with same meaning have same graphs
			- *frame representations* - semantic networks where nodes have structure
				- ex. each frame has age, height, weight, ...
			- when agent faces *new situation* - slots can be filled in, may trigger actions / retrieval of other frames
			- inheritance of properties between frames
			- frames can contain relationships and procedures to carry out after various slots filled