---
layout: notes
title: cs theory
category: cs
---

#  cs theory

Some notes on theoretical computer science, based on UVA's course.

## introduction
- Chomsky hierarchy of languages: $L_3 \subset L_2 \subset L_1 \subset L_R \subset L_0 \subset Σ*$
  - each L is a set of languages
  - $L_0=L_{RE}$ - unrestricted grammars - general phase structure grammars - recursively enumerable languages - include all formal grammars. They generate exactly all languages that can be recognized by a Turing machine.
    - computable, maybe undecidable (if not in L_R)
  - L_R - recursive grammars - Turing machine that halts eventually
    - decidable
  - L_1 - context-sensitive grammars - all languages that can be recognized by a linear bounded automaton
  - L_2 - context-free grammars - these languages are exactly all languages that can be recognized by a non-deterministic pushdown automaton.
  - L_3 - regular grammars - all languages that can be decided by a finite state automaton
    - contains Σ*, $\vert Σ*\vert $ is countably infinite
- strings
- languages
  - Σ* Kleene Closure has multiple definitions
    - {w $\vert $ w is a finite length string ^ w is a string over Σ}
    - {xw $\vert $ w in Σ* ^ x in Σ} U {Ɛ}
  - Σ_i has strings of length i
- problems
- automata
  - delta v delta-hat - delta hat transitions on a string not a symbol
  - $\vert $- notation writes the state between the symbols you have read and have yet to read
  - $\vert $- notation with * writes the state before the symbols you have to read and after what you have read
- grammars
  - leftmost grammar - expand leftmost variables first - doesn't matter for context-free
  - parse tree - write string on bottom
- sets
  - finite
  - countably infinite
  - not countably infinite
- mappings
  1. onto - each output has at least 1 input
  2. 1-1 - each output has at most 1 input
  3. total - each input has at least 1 output
  4. function - each input has at most 1 output
  - equivalence relation - reflexive, symmetric, transitive
- proof methods
  - **read induction **
- library of babel
  - distinct number of books, each contained, but infinite room

## ch 1-3 - finite automata, regular expressions
- alphabet - any nonempty finite set
- string - finite sequence of symbols from an alphabet
- induction hypothesis - assumption that P(i) is true
- lexicographic ordering - {Ɛ,0,1,00,01,10,11,000,...}
- finite automata - like a Markov chain w/out probabilities - 5 parts
  1. states
  2. E - finite set called the alphabet 
  3. f: Q x E -> Q is the transition function
    - ex. f(q,0) = q'
  4. start state
  5. final states
- language - L(M)=A - means A is the set of all strings that the machine M accepts
- A* = {$x_1x_2...x_k \vert  k\geq0 \wedge x_i \in A$}
- A+ = A* - Ɛ
- concatenation A o B = {xy $\vert $ x in A and y in B}
- regular language - is recognized by a finite automata
  - class of regular languages is closed under union, concatenation, star operation
  - nondeterministic automata
    - can have multiple transition states for one symbol
    - can transition on Ɛ
    - can be thought of as a tree
    - After reading that symbol, the machine splits into multiple copies of itself and follows all the possibilities in parallel. Each copy of the machine takes one of the possible ways to proceed and continues as before. If there are subsequent choices, the machine splits again. 
    - If the next input symbol doesn't appear on any of the arrows exiting the current state, that copy of the machine dies. 
    - if any copy is in an accept state at the end of the input, the NFA accepts the input string.
  - can also use regular expressions (stuff like unions) instead of finite automata 
    - to convert, first convert to gnfa
  - gnfa (generalized nfa) - start state isn't accept state
- nonregular languages - isn't recognized by a finite automata
  - ex. C = {w $\vert $ w has an equal number of Os and 1s}
  - requires infinite states

## ch 4 - properties of regular languages (except Sections 4.2.3 and 4.2.4) 
- pumping lemma- proves languages not to be regular
- if L regular, there exists a constant n such that for every string w in L such that \vert w\vert  ≥ n, we can break w into 3 strings w=xyz, such that:
  1. y≠Ɛ
  2. $\vert xy\vert $ ≤ n
  3. For all k ≥ 0, x y^k z is also in L
- closed under union, intersection, complement, concatenation, closure, difference, reversal
- convert NFA to DFA - write the possible routes to the final state, write the intermediate states, remove unnecessary ones
- minimization of DFAs
  - eliminate any state that can't be reached
  - partition remaining states into blocks so all states in same block are equivalent
    - can't do this grouping for nfas

## ch 5 - context free grammars and languages
- $w^R$ = reverse
- context-free grammar - more powerful way to describe a language
- ex. substitution rules (generates 0#1)
  - A -> OA1
  - A -> B
  - B -> #
- def
  1. variables - finite set
  2. terminals - alphabet
  3. productions 
  4. start variable
- recursive inference - start with terminals, show that string is in grammar
- derivation - sequence of substitutions to obtain a string
  - can also make these into parse trees
- leftmost derivation - at each step we expand leftmost variable
- arrow with a star does many derivations at once
- parse tree - final answer is at bottom
- sentential form - the string at any step in a derivation
- proofs in 5.2s
- w equivalence
  1. parse tree
  2. leftmost derivation
  3. rightmost derivation
  4. recursive inference
  5. derivation
- if else grammar: $S \to \epsilon \vert  SS \vert  iS \vert  iSeS $
- context-free grammars used for parsers (compilers), matching parentheses, palindromes, if-else, html, xml
- if a grammar generates a string in several different ways, we say that the string is derived ambiguously in that grammar
- ambiguity resolution
  1. some operators take precedence
  2. make things left-associative
  - think about terms, expressions, factors
  - if unambiguous, leftmost derivation will be unique
- in an unambiguous grammar, leftmost derivations will be unique
- inherently ambiguous language - all its grammars are ambiguous
  - ex: $L = {a^nb^nc^md^m} \cup {a^nb^mc^md^n} , n \geq 1, m\geq1$

## ch 6 - pushdown automata (don't need to know 6.3 proofs)
- pushdown automata - have extra stack of memory - equivalent to context-free grammar
  - similiar to parser in typical compiler
- two ways of accepting
  1. entering accept state
  2. accept by emptying stack
  - convert from empty stack to accept state
    - add symbol X_1
    - start by pushing it onto the stack then push on Z_1, spontaneously transition to q_0
    - everything has epsilon-transition to final accepting state when they read X_1
  - convert accept state to empty stack
    - add symbol X_1 under Z_1 (this is so we never empty stack unless we are in p- there are no transitions on X_1)
    - all accept states transition to new state p
    - p epsilon-transitions to itself, removes element from each stack every time
- 6.3
  - convert context free grammar to empty stack
    - simulate leftmost derivations
    - put answer on stack, most recent variable on top
    - if terminal remove
    - if variable nondeterministically expand
    - if empty stack, accept
  - convert PDA to grammar
    - every transition is of from pXq
    - variables of the form [pXq] (X is on the stack)
      - [pXq] -> a where a is what transitioned p to q
- pushdown automata can transition on epsilon
- def:
  1. transition function - takes (state,symbol,stack symbol) - returns set of pairs (new state, new string to put on stack - length 0, 1, or more)
  2. start state
  3. start symbol (stack starts with one instance of this symbol)
  4. set of accepting states
  5. set of all states
  6. alphabet
  7. stack alphabet
- ex. palindromes
  1. push onto stack and continue OR
  2. assume we are in middle and start popping stack - if empty, accept input up to this point
- label diagrams with i, X/Y - what input is used and new/old tops of stack
- ID for PDA: (state,remaining string,stack)
  - conventionally, we put top of stack on left
- parsers generally behave like deterministic PDA
- DPDA also includes all regular languages, not all context free languages
  - only include unambiguous grammars

## ch 7 - properties of CFLs

- Chomsky Normal Form
  1. A->BC
  2. A->a
    - no epsilon transitions		
    - for any variable that derived to epsilon (ex. A -*> epsilon)
      - if B -> CAD
      - replace with B -> CD and B -> CAD and remove all places where A could become epsilon
  - no unit productions
  - eliminate useless symbols
  - works for any CFL
- Greibach Normal Form
  1. A->aw where a is terminal, w is string of 0 or more variables
  2. every derivation takes exactly n steps (n length)
- generating - if x produces some terminal string w
- reachable - x reachable if S ${\to}^*$ aXb for some a,b 
- CFL pumping lemma - pick two small strings to pump
- If L CFL, then $\vert z\vert  \geq n$, we can break z into 5 strings z=uvwxy, such that:
  1. vx ≠ Ɛ
  2. $\vert vwx\vert  \leq n$, middle portion not too long
  3. For all i ≥ 0, $u v^i w x^i y \in$ L
  - ex. $\{0^n1^n\}$
  - often have to break it into cases
  - proof uses Chomsky Normal Form
- not context free examples
  - $\{0^n1^n2^n\vert n\geq1\}$
  - {$0^i1^j2^i3^j\vert i\geq 1,j\geq 1$}
  - {ww$\vert w \in \{0,1\}^*$ }
- closed under union, concatenation, closure, and positive closure, homomorphism, reversal, inverse homomorphism, substitutions
  - intersection with a regular language (basically run in parallel)
- not closed under intersection, complement
- substitution - replace each letter of alphabet with a language
  - s(a) = $L_a$
  - if $w = ax$, $s(w) = L_aL_x$
  - if L CFL, s(L) CFL
- time complexities
  - O(n)
    - CFG to PDA
    - PDA final state -> empty stack
    - PDA empty stack -> final state
  - PDA to CFG: O($n^3$) with size O(n^3)
  - converstion to CNF: O(n^2) with size O(n^2)
  - emptiness of CFL: O(n)
- testing emptiness - O(n)
  - which symbols are reachable
- test membership with dynamic programming table - O(n^3)
  - CYK algorithm

## ch 8 - intro to turing machines (except 8.5.3)
- Turing Machine def
  1. states
  2. start state
  3. final states
  4. input symbols
  5. tape symbols (includes input symbols)
  6. transition function $\delta(q,X)=\delta(q,Y,L)$
  7. B - blank symbol
    - infinite blanks on either side
- arc has X/Y  D with old/new tape symbols and direction
- if the TM enters accepting state, it accepts
   - assume it halts if it accepts
- we can think of Turing machine as having multiple tracks (symbol could represent a tuple like [X,Y])
- multitape TM has each head move independently, multitrack doesn't
  - common use one track for data, one track for mark

- running time - number of steps that TM makes
- NTM - nondeterministic Turing machine - accepts no languages not accepted by a deterministic TM
- halts if enters a state q, scanning X, and there is no move for (q,X)
- restrictions that don't change things
  - tape infinite only to right
  - TM can't print blank
- simplified machines
  - two stacks machine - one stack keeps track of left, one right
  - every recursively enumerable language is accepted by a two-counter machine
- TM can simulate computer, and time is some polynomial multiple of computer time (O(n^3))
  - limit on how big a number computer can store - one instruction - word can only grow by 1 bit
- LBA - linear bounded automaton - Turing machine with left and right end markers
- programs might take infinitely long before terminating - can't be decided
- turing machine can take 2 inputs: program P and input I
- ID - instantaneous description      
  - write $X_1X_2...qX_iX_{i+1}...$ where q is scanning X_i
- program that prints "h" as input -> yes or no
  - imagine instead of no prints h
  - now feed it to itself
    - if it would print h, now prints yes - paradox! therefore such a machine can't exist
- TM simulating computer
  1. tape that has memory
  2. tape with instruction counter
  3. tape with memory address
- reduction - we know X is undecidable - if solving Y implies solving X, then Y is undecidable
  - if X reduces to Y, solving Y solves X
  - define a total mapping from X to Y
    - $X \leq _m Y$ - X reduces to Y - mapping reduction, solving Y solves X
- intractable - take a very long time to solve (not polynomial)
- <> notation means bitstring representation
- $<n> = 0^n$
- $<m,w> means w \in L(M)$
- KD - "known to be distinct"
- idempotent - R + R = R

## ch 9 - undecidability (9.1,9.2,9.3)
- does this TM accept (the code for) itself as input?
- enumerate binary strings - add a leading 1
- express TM as binary string
  - give it a number
  - TM uses this for each transition
    - separate transitions with 11
- diagonalization language - set of strings w_i such that w_i is not in L(M_i)
  - make table with M_i as rows, w_j as cols
  - complement the diagonal is characteristic vector in L_d
    - diagonal can't be characteristic vector of any TM
  - not RE
- recursive - complement is also recursive
  - just switch accept and reject
- if language and complement are both RE, then L is recursive
- universal language - set of binary strings that encode a pair (M,W) where M is TM, w $\in (0+1)^*$ - set of strings representing a TM and an input accepted by that TM
- there is a universal Turing machine such that L_u = L(U)
  - L_u is undecidable: RE but not recursive
- halting problem - RE but not recursive
- Rice's Thm - all nontrivial properties of the RE languages are undecidable
  - property of the RE languages is a set of RE languages
  - property is trivial if it is either empty or is all RE languages
    - empty property $\emptyset$ is different from the property of being an empty language {$\emptyset$}
  - ex. "the language is context-free, empty, finite, regular"
- however properties such as 5 states are decidable

## Ch 10 - 10.1-10.4 know the additional problems that are NP-complete 
- intractable - can't be solved in polynomial time
- NP-complete examples
  1. boolean satisfiability
    1. symbols ^-, etc. are represent by themselves
    2. x_i is represented by x followed by i in binary
    - Cook's thm - SAT is NP-complete
      1. show SAT in NP
      2. show all other NP reduce to SAT
      - pf involves matrix of cell/ID facts
        - cols are ID 0,1,...,p(n)
        - rows are alpha_0,alpha_1,...alpha_p(n)
        - for any problem's machine M, there is polynomial-time-converter for M that uses SAT decider to solve in polynomial time
  2. 3SAT - easier to reduce things to
    - AND of clauses each of which is the OR of exactly three variables or negated variables
    - conjunctive normal form - if it is the AND of clauses
    - conversion to cnf isn't always polynomial time - don't have to convert to equivalent expression, just have to both be satisfiable at the same times
      1. push all negatives down the expression tree - linear
      2. put it in cnf - demorgans, double negation
    - literal - variable or a negated variable
    - k-conjunctive normal form - k is number of literals in clauses
  3. traveling salesman problem - find cycle of weight less than W
    - O(m!)
  4. Independent Set - graph G and a lower bound k - yes if and only if G has an indpendent set of k nodes
    - none of them are connected by an edge
    - reduction from 3SAT
  5. node-cover problem
    - node cover - every node is on one of the edges 
  6. Undirected Hamiltonian circuit problem
    - TSP with all weights 1
  7. Directed Hamiltonian-Circuit Problem
  8. subset sum
    - is there a subset of numbers that sums to a number 
- reductions must be polynomial-time reductions
- P - solvable in polynomial by deterministic TM
- NP - solvable in polynomial time by nondeterministic TM
  - NP-completeness (Karp-completeness) - a problem is at least as hard as any problem in NP = for every language L' in NP, there is a polynomial-time reduction of L' to L
  - Cook-completeness equivalent to NP-completeness - if given a meachansim that in one unit of time would answer any equestion about membership of a string in P, it was possible to recognize any language in NP in polynomial time
  - NP-hard - we don't know if L is in NP, but every problem  in NP reduces to L in polynomial time
- if some NP-complete problem p is in P then P=NP
- there are things between polynomial and exponential time (like 2^logn), and we group these in with the exponential category
- could have P polynomials run forever when they don't accept
  - could simply tell them to stop after a certain amount of steps
- there are algorithms called verifiers

## more on NP Completeness
- a language is polynomial-time reducible to a language B if there is a polynomial time comoputable function that maps one to the other
- to solve a problem, efficiently transform to another problem, and then use a solver for the other problem
- satisfiability problem - check if a boolean expression is true
  - have to test every possible boolean value - 2^n where n is number of variables
    - this can be mapped to all problems of NP
    - ex. traveling salesman can be reduced to satisfiability
- P - set of all problems that can be solved in polynomial time
- NP - solved in polynomial time if we allow nondeterminism
  - we count the time as the length of the shortest path
- NP-hard problem L'
  1. every L is NP reduces to L' in polynomial time
- NP-complete L'
  1. L' is NP-hard
  2. L' is in NP
  - ex. graph coloring
  - partitioning into equal sums
- if one NP-complete problem is in P, P=NP
- decider vs. optimizer
  - decider tells whether it was solved or not
  - if you keep asking it boolean questions it gives you the answer
- graph clique problem - given a graph and an integer k is there a subgraph in G that is a complete graph of size k
  - this is reduction from boolean satisfiability
- graph 3-colorability
  - reduction from satisfiability - prove with or gate type structure
- approximation algorithms
  - find minimum
    - greedy - keep going down
    - genetic algorithms - pretty bad
  - minimum vertex cover problem - given a graph, find a minimum set of vertices such that each edge is incident to at least one of these vertices
    - NP-complete
    - can not be approximated within 1.36*solution 
    - can be approximated within 2*solution in linear time
      - pick an edge, pick its endpoints
      - put them in solution
      - eliminate these points and their edges from the graph
      - repeat
- maximum cut problem - given a graph, find a partition of the vertices maximizing the number of crossing edges
  - can not be approximated within 17/16*solution
  - can be approximated within 2*solution
    - if moving arbitrary node across partition will improve the cut, then do so
    - repeat