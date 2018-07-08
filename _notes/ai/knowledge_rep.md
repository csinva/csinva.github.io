---
layout: notes
section-type: notes
title: representations
category: ai
---

* TOC
{:toc}
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
      - semantic networks - objects and relationships between them		- like is a, can, has
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