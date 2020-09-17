# evaluating AI decision support tools as embedded systems

**sina fazelpour, cmu**

- overview
  - approach interpretability from functional, goal-oriented perspective
    - focus shouldn't be on prediction but on actual decision-making task (with human in the loop)
  - current focus
    - locus of evaluation - usually the model
    - locus of intervention - usually the algorithm
- do **proxy outcomes** reflect the real setting properly?
  - need to consider organizational + societal challenges to interpretability
- fairness - aggregate metrics can be fair/accurate but differ for different subgroups
- **complementarity**
  - best individual performers do not generally constitute the best teams
  - want AI / ppl to be good at different things
  - optimize for collaboration - learn when to defer to experts
    - predict responsibly - improving fairness and accuracy by learning to defer
    - learning to complement humans
- 2 dimensions for how social sources are integrated into individual decision-making
  - **informational dimension** - desire to be right in uncertain decision environments
    - perceive reliability, history of personal interactions (can bias people towards algorithm = *automation bias behavior* or away from algorithm = *algorithmic aversion behavior*)
      - humans tend to tolerate more errors coming from humans
  - **incentive dimension** - personal, organizational, or societal
    - solutions to these are rarely technical
      - need to consider interaction between interp methods + user psychology
      - ppl tend to like more complex explanations (e.g. neurips papers, complex eqs) and sometimes dismiss simple explanations
      - ideal expert can adhere to model when it works and ignore it when it doesn't
- need to monitor things during deployment
  - ex. goodhart's law - things that were good signals become poor when they are artificially gamed
    - regularly retraining can help fix this
      - sometimes need backward compatibility to maintain trust
- actionable insights
  - task - shouldn't be predicting when humans do well, should be on complementarity