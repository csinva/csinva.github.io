# 7 - propositional logic

- declarative vs procedural (knowing how to ride a bike)
- horn clause - at most one positive
  - definite clause - exactly one positive
  - goal clause - 0 positive
- TT-ENTAILS: check everything
- forward-chaining/backward chaining
- DPLL: TT-ENTAILS with 3 improvements
  - early termination
  - pure symbols
  - unit clause
- agents
  - use A* with entailment
  - SATPLAN
- propositional logic: facts - true/false/unknown

# 8 - first-order logic

- first-order logic: add objects, relations, quantifiers ($\exists, \forall$)

# 9 - inference in first-order logic

- unification
- first-order logic forward-chaining: FOL-FC-ASK
  - efficient forward chaining
    - conjunct ordering
    - ignore redundanat rules
    - ignore irrelevant facts w/ backward chaining
- backward-chaining: FOL-BC-ASK
  - generator - returns multiple times

# 10 - classical planning

# 12 - knowledge rep