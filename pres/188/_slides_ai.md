---
title: ai slides
separator: '----'
verticalSeparator: '---'
highlightTheme: ir-black
typora-copy-images-to: ./assets_files
revealOptions:
    transition: 'slide'
	transitionSpeed: 'fast'
---

<h1> artificial intelligence </h1>
*press esc to navigate slides*

**<a href="https://twitter.com/csinva_">@csinva_</a>**

[![](assets_files/GitHub-Mark-64px.png)](https://github.com/csinva/csinva.github.io/blob/master/_slides/ai_slides/slides.md)

# uninformed search

## the setup

1. initial state: $s_0$
2. actions at each state: go($s_1$), go($s_2$)
3. transition model: result($s_0$, go($s_1$)) = $s_1$
4. goal states: $s_\text{goal}$
5. path cost function: cost($s_0$, a, $s_1$)

## how to search

- TREE-SEARCH - continuously expand the frontier
  - *frontier*  = set of all leaf nodes available to expand
- GRAPH-SEARCH - also keep set of visited states

## metrics

- *optimal* - finds best solution
- *complete* - terminates in finite steps
- *time, space complexity*
  - *search cost* - just time/memory
- *total cost* - search cost + *path cost*

##  summary

![](assets_files/uninformed_comparisons.png)

# A* & Heuristics

## informed search

- *informed search* - use $g(n)$ and $h(n)$
  - *g(n)*:  cost from start to n
  - *heuristic h(n)*: cost from n to goal
- *best-first* - choose nodes with best f
  - f=g: uniform-cost search
  - f=h: greedy best-first search
- $A^*$: $f(n) = g(n) + h(n)$

## admissible

- *admissible*: $h(n)$ never overestimates
- $\implies A^*$ (with tree search) is optimal and complete

![graph](assets_files/graph.png)


## consistent

- *consistent*: $h(n) \leq cost(n \to n') + h(n')$
  - monotonic
- $\implies A^*$ (with graph search) is optimal and complete 

## optimally efficient

- *consistent* $\implies$ *optimally efficient* (guaranteed to expand fewest nodes) 
  - never re-open a node
- weaknesses
  - must store all nodes in memory

## memory-bounded heuristic search

- *iterative-deepening* $A^*$ - cutoff f-cost
- *recursive best-first search*
  - each selected node backs up best f alternative
  - if exceeding this, rewind
  - when rewinding, replace nodes with with best child f
- $SMA^*$ - simplified memory-bounded
  -  when memory full, collapse worst leaf

##  recursive best-first

![Screen Shot 2018-06-28 at 7.47.11 PM](assets_files/Screen Shot 2018-06-28 at 7.47.11 PM.png)

## heuristic functions

- want big h(n) because we expand everything with $f(n) < C^*$
  - $h_1$ dominates $h_2$ if $h_1(n) \geq h_2(n) \: \forall \: n$
  - combining heuristics: pick $h(n) = max[h_1(n), ..., h_m(n)]$

- *relaxed problem* yields admissible heuristics

  - ex. 8-tile solver


# local search

## definitions

- *local search* looks for solution not path ~ like optimization
- maintains only *current node* and its neighbors

## discrete space

- *hill-climbing* = *greedy local search* 
  - stochastic hill climbing
  - random-restart hill climbing
- *simulated annealing* - pick random move
  - if better accept
  - else accept with probability $\exp(\Delta f / T_t)$
- *local beam search* - pick k starts, then choose the best k states from their neighbors
  - *stochastic beam search* - pick best k with prob proportional to how good they are

## genetic algorithms

![Screen Shot 2018-06-28 at 8.12.02 PM](assets_files/Screen Shot 2018-06-28 at 8.12.02 PM.png)

*schema* - representation

## continuous space

- could just discretize neighborhood of each state

- SGD: *line search* - double $\alpha$ until f increases
- *Newton-Raphson* method: $x = x - H_f^{-1} (x) \nabla f(x)$![slide_8](assets_files/slide_8.jpg)

# constraint satisfaction problems

## definitions

- variables, domains, constraints
- goal: *assignment* of variables
  - *consistent* - doesn't violate constraints
  - *complete* - every variable is assigned
- 2 ways to solve
  1. local search - assign all variables and then alter
  2. backtracking search (with inference) - assign one at a time

## example
![Screen Shot 2018-06-28 at 8.40.49 PM](assets_files/australia.png)

## 1 - local search for csps
- start with some assignment to variables
- *min-conflicts* heuristic - change variable to minimize conflicts
  - can escape plateaus with *tabu search* - keep small list of visited states
  - could use *constraint weighting*

## 2 - backtracking
- depth-first search that backtracks when no legal values left
  - b1 - variable and value ordering
  - b2 - interleaving search and inference
  - b3 - intelligent backtracking - looking backward

## b1 - variable and value ordering

- *commutative*
- heuristics
  - *minimum-remaining-values*
  - *degree* - pick variable involved in most constraints
  - *least-constraining-value*

## b2 - interleaving search and inference

- *forward checking* - after assigning, check arc-consistency on neighbors
- *maintaining arc consistency (MAC)* - after assigning, arc consistency initialized on neighbors

## constraint graph

*this is not the search graph!*

![Screen Shot 2018-06-28 at 8.40.49 PM](assets_files/Screen Shot 2018-06-28 at 8.40.49 PM.png)

## inference

- **constraint propagation** uses constraints to prune domains of variables
- finite-domain constraint $\equiv$  set of binary constraints w/ auxiliary variables
  - ex. *dual graph transformation*: constraint $\to$ variable, shared variables $\to$ edges

## basic constraint propagation

- *node consistency* - unary constraints
- *arc consistency* - satisfy binary constraints (AC-3 algorithm)
  - for each arc, apply it
    - if things changed, re-add all the neighboring arcs to the set
  - $O(cd^3)$ where $d = \vert domain\vert $, c = num arcs

## advanced constraint propagation

- *path consistency* - consider constraints on triplets - PC-2 algorithm
  - extends to *k-consistency*
  - *strongly k-consistent* - also (k-1), ..., 1-consistent
    - $\implies O(k^2d)$ to solve
- global constraints

## b3 - intelligent backtracking

- *conflict set* for each node (list of variable assignments that deleted things from its domain)
- *backjumping*  - backtracks to most recent assignment in conflict set
- *conflict-directed backjumping* 
  - let $X_j$ be current variable and $conf(X_j)$ be conflict set. If every possible value for $X_j$ fails, backjump to the most recent variable $X_i$ in $conf(X_j)$ and set $conf(X_i) = conf(X_i) \cup conf(X_j) - X_i$
- *constraint learning* - finding min set of assigments from conflict set that causes problem


## structure of problems
- connected components of constraint graph are independent subproblems
- *tree* - any 2 variables are connected by only one path
  - *directed arc consistency* - ordered variables $X_i$, every $X_i$ is consistent with each $X_j$ for j>i
    - tree with n nodes can be made directed arc-consisten in $O(n)$ steps - $O(nd^2)$

## making trees

1. assign variables so remaining variables form a tree
  - assigned variables called *cycle cutset* with size c
  - $O[d^c \cdot (n-c) d^2]$
  - finding smallest cutset is hard, but can use approximation called *cutset conditioning*
2. *tree decomposition* - view each subproblem as a mega-variable
  - *tree width* w - size of largest subproblem - 1
  - solvable in $O(n d^{w+1})$

# game trees

## what are game trees?

- adversarial search problems - adversaries attempt to keep us from goal state

## components

- states have utilities ~ like the score obtained at a state
- state's value is often best possible utility obtainable from a state

## minimax algorithm

- visit nodes in postorder traversal (like DFS)
- at each decision node, pick the node with best utility

![Screen Shot 2019-09-17 at 10.31.15 AM](assets_files/Screen Shot 2019-09-17 at 10.31.15 AM.png)


## alpha-beta pruning

- can stop searching a branch if we know it won't be picked

![alpha-beta-pruning](assets_files/alpha-beta-pruning.jpg)

## evaluation functions

- estimate utility of a node
- often do this as a linear function of features

# mdps (nondeterministic search)

## nondeterministic search

- multiple successors can result from an action (w/ some prob)
- can represent this as a a game tree with expectimax chance nodes
- mdps can help solve these problems
- goal: find a policy $\pi$ which maps states to actions

## elements

- standard: states, actions, start, terminal states
- transition function: $T(s, a, s')$
- reward function: $R(s, a, s')$
- try to maximize sum of rewards or discounted sum of rewards

## mdps

- memoryless: $T(s, a, s') = P(s'|s, a)$
- value $V^*(s)$ = maximum expected utility starting at s
- q-value $Q^*(s, a)$ = maximum expected utility starting in s, taking action a

## the bellman eqn

- $V^*(s) = \underset{a}{\max}\: Q^*(s, a)$
  - $Q^*(s, a) = \underset{s'}{\sum} T(s, a, s')[R(s, a, s') + \gamma V^*(s')]$
- condition for optimality

## value iteration

initialize state values to zero and iteratively them

1. $V_0(s) = 0$
2. Bellman update for each state

- once we have values for each state, $\pi^*(s) = \underset{a}{\text{argmax}} \: Q^*(s, a)$

## policy iteration

initialize policy and iteratively update it

1. define $\pi_0(s)$
2. Bellman update for policy, where values are computed using the current policy $V^\pi_i(s)$

# rl

## what is rl

- don't know rewards / transitions, but want to learn best policy to optimize rewards
- must balance exploration with exploitation
- uses samples or **episodes** = samples until reaching a terminal state

## types of rl

- **model-based** learn transition + rewards and used value/policy iteration vs **model-free**: directly estimate values /q-values
  - keeping counts of transitions can be memory-intensive
- **passive** (given policy - e.g. direct evaluation, TD-learning) and **active** (e.g. Q-learning) needs to learn policy
- **on-policy learning** requires policy to be optimal to learn optimal values while **off-policy learning** can learn optimal values following suboptimal policy (e.g. Q-learning)



## equations

- direct methods: just keep track of counts
- TD (bellman eqn w/ fixed policy): $V^\pi(s) = \underset{s'}{\sum} T(s, \pi(s), s')\underbrace{ [R(s, \pi(s), s') + \gamma V^\pi (s')]}_{\text{sample}}$
  -  exponential moving average: $V^\pi(s) = (1- \alpha) V^\pi (s) + \alpha \cdot sample$
- $Q_{k+1}(s, a) = \underset{s'}{\sum} T(s, a, s') \cdot \underbrace{[R(s, a, s') + \gamma \underset{a'}{\max} Q(s', a')]}_{\text{sample}}$ (like value iteration, but no max)
  -  exponential moving average: $Q(s, a) = (1 - \alpha) Q(s, a) + \alpha \cdot sample$
  - approximate q-learning = use feature-based repr.

## exploration + exploitation

- $\epsilon$-greedy policy - act randomly with probability $\epsilon$
- exploration function - replace Q(s, a) with some function based on how many times state has been visited

# bayesian networks

## variables

1. query variables - unkown and we want to know them
2. evidence variables - known
3. hidden variables - present, but unknown

## representation

- each node is conditionally indep of all ancestors given all of its parents
- **d-separation** - color evidence vars, they separate unles they are at a common effect (or downstream)

## triples

| | Causal chain                                                 | Common cause                                                 | Common effect                                                |
| :-- | --:-- | -:- | -:- |
| | ![Screen Shot 2019-11-05 at 2.42.01 PM](assets_files/Screen Shot 2019-11-05 at 2.42.01 PM.png) | ![Screen Shot 2019-11-05 at 2.42.05 PM](assets_files/Screen Shot 2019-11-05 at 2.42.05 PM.png) | ![Screen Shot 2019-11-05 at 2.42.10 PM](assets_files/Screen Shot 2019-11-05 at 2.42.10 PM.png) |
| $X \perp Z$? |  ❌  |   ❌ | ✅      |
| $X \perp Z \vert Y$? |  ✅  |   ✅ | ❌     |




## inference

- inference by enumeration: sum over all the variables
- elimination: join all factors and sum them out
  - two steps: alternate joining and eliminating
  - want to pick ordering to create smallest factors

## sampling

- **prior sampling**: sample and throw away things which don't match
- **rejection sampling**: throw away anything as soon as it doesn't match
- **likelihood weighting** - fix evidence variables and weight them by their likelihood
- **gibbs sampling** - set variables random and repeatedly resample one var at a time


# decision networks

## what is a decision network?

similar to bayes nets, but we add 2 things:
1. action nodes - we have complete control
2. utility nodes - give us a result

## decision theory

- $EU(a|e) = \sum_{s'} P(s'|s, a) U(s')$
- $MEU(e) = \underset{a}{\max} EU(a|e)$
- $VPI(T) = E_T[MEU(e, t)] - MEU(e)$

# hidden markov models

## 4 inference problems

1. *filtering* = *state estimation* - compute `$P(W_t | e_{1:t})$`

2. *prediction* - compute `$P(W_{t+k}|e_{1:t})$` for $k>0$

3. *smoothing* - compute `$P(W_{k}|e_{1:t})$` for $0 < k < t$

4. *most likely explanation* - `$\underset{w_{1:t}}{\text{argmax}}\:P(w_{1:t}|e_{1:t})$`

## simple markov model

![Screen Shot 2019-11-06 at 5.00.18 PM](assets_files/Screen Shot 2019-11-06 at 5.00.18 PM.png)

`$P(W_{i+1}) = \underset{w_i}{\sum} P(W_{i+1}|w_i) P (w_i)$`

## hidden markov model

![Screen Shot 2019-11-06 at 4.28.20 PM](assets_files/Screen Shot 2019-11-06 at 4.28.20 PM.png)

`$$\underbrace{P(w_{t+1}|e_{1:t+1})}_{\text{new state}} = \alpha \: \underbrace{P(e_{t+1}|w_{t+1})}_{\text{sensor}} \cdot \underset{w_t}{\sum} \: \underbrace{P(w_{t+1}|w_t)}_{\text{transition}} \cdot \underbrace{P(w_t|e_{1:t})}_{\text{old state}}$$`

## viterbi algorithm

`$\underbrace{\underset{w_{1:t}}{\text{max}} \: P(w_{1:t}, W_{t+1}|e_{1:t+1})}_{\text{mle}} =\\ \alpha \: \underbrace{P(e_{t+1}|W_{t+1})}_{\text{sensor}} \cdot \underset{w_t}{\text{max}} \left[ \: \underbrace{P(W_{t+1}|w_t)}_{\text{transition}} \cdot \underbrace{\underset{w_{1:t-1}}{\text{max}} \:P(w_{1:t-1}, w_{t+1}|e_{1:t})}_{\text{max prev state}} \right]$`

## particle filtering

sample a bunch of particles and use to approximate probabilities:

1. initialize particles
   1. time-lapse update
   2. observation update

speed-ups when num particles < num possible states

# extra topics








# propositional logic

- declarative vs procedural (knowing how to ride a bike)
- horn clause - at most one positive
  - definite clause - exactly one positive
  - goal clause - 0 positive

## TT-ENTAILS: check everything
- forward-chaining/backward chaining

## DPLL: TT-ENTAILS with 3 improvements
  - early termination
  - pure symbols
  - unit clause

## agents
  - use A* with entailment
  - SATPLAN
- propositional logic: facts - true/false/unknown

# first-order logic

![Screen Shot 2018-08-01 at 7.52.25 PM](assets_files/Screen Shot 2018-08-01 at 7.52.25 PM.png)


- first-order logic: add objects, relations, quantifiers ($\exists, \forall$)

  - unification

## inference 1

- simple: first-order logic forward-chaining: FOL-FC-ASK

  - efficient forward chaining
    - conjunct ordering
    - ignore redundanat rules
    - ignore irrelevant facts w/ backward chaining

## inference 2

- backward-chaining: FOL-BC-ASK
  - generator - returns multiple times

# classical planning

- requires 4 things (like search w/out path cost function)
  - initial state
  - actions
  - transitions
  - goals

## action example


- $Action(Fly(p, from, to))$:
	- PRECOND: $At(p, from) \land Plane(p) \land Airport(from) \land Airport(to)$
	- EFFECT: $\neg At(p, from) \land At(p, to)$
  - can only use variables in the precondition

## planning as search

![search](assets_files/search.jpg)

## algorithms

1. forward state-space search
   1. inefficient, but preferred because good heuristics
2. backward
   1. start with a set of things in the goal (and other fluents can have any value)

## heuristics

- ex. ignore preconditions
- ex. ignore delete lists - remove all negative literals
- ex. **state abstractions** - many-to-one mapping from states $\to$ abstract states
  - ex. ignore some fluents

# knowledge repr.

- compositionality
- intrinsic / extrinsic properties
- event calculus  

# decisions + rl 

**this re-iterates the relevant equations using the equations in Russel & Norvig (all based on utitily funciton U(s)**

## mdps

$\pi(s) = \underset{\pi}{\text{argmax}} \:U^\pi (s)$

- value iteration: $U(s) = R(s) + \gamma MEU(s)$ (Bellman eqn)
- policy iteration: $U(s) = R(s) + \gamma EU(s)$

## passive rl

*given $\pi$, find $U(s)$*

- ADP: find $P(s'|s, a)$, $R(s) \to $ plug into Bellman eqn
- TD: when $s \to s'$: $U(s) = U(s) + \alpha[R(s) + \gamma U(s') - U(s)]$



## active rl

*find $\pi$, maybe maximize rewards along the way*

*here only Q-learning: $U(s) = \underset{a}{\max}Q(s, a)$*

- ADP: $Q(s, a) =  R(s) + \alpha[\sum_{s'} P(s'|s, a)\underset{\alpha}{\max}Q(s', a')]$
- TD: when $s\to s'$: $Q(s, a) = Q(s, a) + \alpha[R(s) + \gamma \underset{a}{\max}Q(s', a') - Q(s, a)]$
- SARSA: when $s\to s'$: $Q(s, a) = Q(s, a) + \alpha[R(s) + \gamma Q(s', a') - Q(s, a)]$