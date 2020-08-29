# why causal interpretations matter for algorithmic bias mitigation: a legal perspective

**alice xiang, partnership on AI**

- **major pushback:** - since it is so hard to define fairness, maybe unawaress is the best we can do practically?
- partnership on AI - consortium of tech companies
- "reconcilling legal and technical approaches to algorithmic bias" - forthcoming law review
- key issue: extent to which well-meaning algo developers can use protected class variables
  - technical necessity for using protected class variables
  - legal preference fore methods that are blind or neutral to protected class attributes
- why is legal compatibility important
  - to demonstrate evidence of algorithmic bias, fairness metrics must conform to what judges, juries, or regulators would accept as discrimination evidence
  - to deploy bias mitigation, can't conflict with existing law
    - ml fairness approaches might be illegal
- removing protected class variables does not work
  - including protected class variables can help mitigate bias
- 2 perspectives: courts see to learn towards first
  - anti-classification - classification that differs based on protected class attributes is discriminatory
    - in the US, **we can't use protected class variables** (e.g. race, sex, age, disability, national origin, religion)
      - not socieconomic class / geography
      - race/sex are treated symmetrically (e.g. white/black treated same)
  - anti-subordination - law should seek to dismantle hierarchies between protected class groups, even if it requires consciously knowing the protected attribute
    - legal perspective focuses on causality
      - **discrimination is define as making a decision "because of X"**
      - are you strengthening or weakening the causal connection?
      - allows for distinction between bias from historical data and bias from model
    - hard to get actual counterfactual, focus intead on **perceived immutable characteristics** (e.g. greiner & rubin)
- regend of UC v Bakke (1978)
  - race can be used as one of many factors but not directly
  - ruled it was okay to aim for diversity, but not to overturn historical discrimination
- disparate treatment - intentional discrimination
  - burden-shifting framework makes it harder for plaintiffs