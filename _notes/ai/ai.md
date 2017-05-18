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

# classical search
- BFS
- DFS
	- memory efficient - don't have to store all possible paths
	- will find some path very quickly
- depth-limited search - only dfs up to some max_depth
- *iterative deepening* - depth-limited search with growing max_depth
- *heuristic* - estimate of the cost to the goal node will be (from each node)
	- *open* list - keeps track of promising nodes
	- *closed* list - nodes already checked
	- greedy search - best first where best is picked by heuristic h(x) - guess on distance to goal!

### A*
- evaluation f(N) = g(N) + h(N)
	- g(N) is the cost of the best path found so far to N (Djikstra just uses this)
	- h(N) is an admissible heuristic (best first just uses this)
		- h(N) = 0 -> BFS
- *admissible* - constraints on h(N)
	- let h*(N) be true cost from node N to goal
	- *optimistic* h(N) ≤ h*(N)
	- if h(n) admissible, A* is optimal
- *consistent*  -> admissible
	1. satisfies triangle inequality: h(N) ≤ dist(N,N') + h(N')
	2. h(Goal) = 0 
	- implies f is non-decreasing along any path
	- if h is consistent, when A* expands a node, it has already found optimal path to state associated with the node
- if h1(N) ≤ h2(N) for all N and h1 and h2 both admissible,consistent
	- h2 is *more informed* than h1

# local search
- don't care about solution path, just solution state
	- usually no start state
- *greedy* - start randomly, keep going up until you find maximum
- *random walk*: start somewhere, do a random walk
- *hill climbing*: go to local maximum
- *simulated annealing*: more random in beginning (when hot) and more hill climbing at end (cooled down)
	- use sigmoid function to select which solution to use
- genetic algorithms - genotype space gets randomly mutated and evaluated with fitness function
	- need diversity
	- *aritv* - number of inputs
		- aritv=1 - mutation operator
		- aritv=2 - crossover
		- aritv ≥2 - recombination

# game playing search
- game - well-defined problem that requires intelligence
- want to represent the game as a search tree
- Tic-Tac-Toe - *minimax procedure*
	- give leaf nodes a score (1 if AI wins, -1 if AI loses, 0 if a draw)
	- recur backwards
	- usually intractable
- if game is random, use *expectimax procedure*
- alpha-beta pruning - optimization for minimax
	- if already have a winning path, don't calculate others
	- example of *branch-and-bound* algorithm - bound the solution with the best path already discovered
- chess
	- *dynamic* states - 1 or 2 turns, big turning point could occur
	- *quiescent* - no important pieces will be taken soon
	- *horizon problem* - you search to depth d, and moves at d+1, d+2 lead to your queen getting taken
- *terminal state* - state where the game is over
- *utility function* - determines payout of terminal states
- *ply* - each layer in the search tree

# other uncertainty models
### markov model
- *stationarity* - probabilities don't change as a function of t
- transition matrix acts on probability vector
- absorbing MC
	1. has absorbing state
	2. must always be able to get to absorbing state from any other

### hidden markov model
- underlying state of the system is hidden

### markov decision processes
- want to model agents that observe the world, react to what they see
- MDP contains
	- set of possible word states S
	- set of possible actions A
	- real-valued *reward function R(s,a)*
	- solution is policy $\pi (s,a)$
	- value function $V_\pi (s)$ for each state - sums using discount factor
		 - $V = max_a [Q_\pi  (s,a)]$ - value functions using particular action
- optimizing MDP
	- could sum rewards, but results are infinite
	- instead define objective function (maps infinite sequences of rewards to single real numbers)
		- ex. set a finite horizon and sum rewards
		- ex. discounting to prefer earlier rewards (most common)
			- could discount reward n steps away by r^n, 0<r<1
		- ex. average reward rate in the limit
- *value iteration* - takes an MDP and calculates an optimal policy
	- start with arbitrary value function
	- recalculate several times

### partially observable markov decision processes (POMDP)
- agent is not sure what state it's in
- have belief space for what state you are in
	- changes based on observations
- use belief states as the states of an MDP and solve as before
- belief space is continuous [0,1] so we represent it as piecewise linear, and store these discrete lines in memory
- solving with all these lines is difficult

# Bibliography
{ bibliography --cited }