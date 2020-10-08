---
layout: notes
title: representations
category: ai
---

{:toc}

Some notes on knowledge representation based on Berkeley's CS 188 course and  "Artificial Intelligence" Russel & Norvig 3rd Edition

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
  
- statistical

  - distributed - usually different from sparse code (sparser generally less robust)
    - opposite of sparse code = dense code
    - have to check multiple indexes
    - penti's work: distributed
    - usually want these to be robust
    - nlp is main place where unsupervised pretraining widely used
  - hierarchical

  - good representations - *linearly separable*
  - representation that *factors*
  - information bottleneck method: want simple representation that keeps class but throws away lots of extraneous info

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

# Godel, Escher, Bach
**Douglas Hofstadter, 1979**

## meta

- *strange loop* = paradox - self-referential
- zen enlightenment
  - goal: transcend **dualism** = division into concepts (perception, words do this)
  - words give you some truth but always fail to describe some parts of the truth

## music

- **canons** - repeat w/ subtle changes (e.g. pitch shift)
- **fugue** - repeat w/ more substantial changes

## ai

- essential abilities - **can we do these things unsupervised?**
  - to recognize the relative importance of different elements of a situation
  - to find similarities between situations despite differences which may separate them;
  - to draw distinctions between situations despite similarities which may link them
  - to synthesize new concepts by taking old concepts and putting them together in new ways
- intelligence consists of rules at different levels
  - "just plain" rules - like reflexes which respond to stereotyped situations
  - metarules - when situations are mixtures of steretoyped situations, requires rules for deciding which "just plain" rules to apply
  - rules for inventing new rules - when situations can't be classified
    - rules may have to change themselves
- messages - comparison of DNA to a jukebox
  - where is info stored? records? buttons? smashed buttons?
  - what could constitute a Rosetta stone for DNA codes?
  - 3 parts
    - frame - tells you that this is a message
    - outer - tells you how to read a message (e.g. language, style)
    - inner - actual content
  - if decoding is universal, we might call the outer message (e.g. the trigger) the message
- memory - same bits can be used for different things - part of each message specifies the instruction type

## brain

- intelligence involves a calculus of descriptions = symbols
  - symbols represent both classes + instances (maybe both depending on amount of activation / context) - def need some context
  - can have links to other symbols (+priors on these)
  - *top-down logical structure*??
  - different ways to combine symbols get blurry
    - symbols can be learned to branch, merge
  - can harness temporal firing rates to encode more
  - can grow incrementally (greedily)
- analogy of thoughts as trips on a (poorly fleshed out) map

## interpretation

- ex. top is decimal expansion of the sum of the second ($\pi/4$)
  - 7, 8, 5, 3, 9, 8, 1, 6, ...
  - 1, -1/3, +1/5, -1/7, +1/9, -1/11...

## math / logic

- **godel's thm** - limitation of any formal axiomatic system: cannot make a program to find a complete + consistent set of axioms
- **church-turing thesis** - a function on the natural numbers can be calculated by an effective method if and only if it is computable by a Turing machine
  - no system can do computation which cannot be broken down into simple elements
- **decision procedure** - decides whether something is a theorem - must terminate
- we can think of theorems as strings in a formal system
- **interpretation** - correspondence between symbols and words
  - ideally, these are meaningful isomorphisms between codes and reality
  - not all interpretations imply meaningful (or valid) corresponding codes
  - there might be multiple, equally valid interpretations
  - consistency depends on interpretation:
  - **consistency** - when every theorem, upon interpretation, comes out true (in some imaginable world)
  - **completeness** - when all statements which are true (in some imageinable world), and which can be expressed as well-formed strings of the system, are theorems
- slightly different axioms lead to elliptical/hyperbolic geometry instead of Euclidean geometry
- godel numbering - can replace all symbols w/ numbers and all typographic rules w/ arithmetic rules
- 2 key idesas
  - strings can speak about other strings
  - self-scrutiny can be entire concentrated into a single string
- every aspect of thinking can be viewed as a high-level description of a system which, on a low level, is governed by simple, even formal rules

## causality

- what counterfactuals are the most realistic
  - different things are stable at different levels

## biology

- dna -> rna -> proteins = sequence of amino acids
  - folds w/ valrious levels of structure (like music)
- self-rep - what counts?
  - quine? instructions on jukebox? human reproduction?