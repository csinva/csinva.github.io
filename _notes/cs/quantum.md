- what does physics tell us about the limits of computers?
- NP - can check soln in polynomial time
- NP-hard - if solved, solves every NP
- NP-complete - NP hard and in NP
- *church-turing thesis* $\implies$ turing machine polynomial time should be best possible in the universe
  - physics could allow us to do better than a turing machine
- examples
  - glass plates with soapy water - forms minimum steiner tree
  - can get stuck in local optimum
  - ex. protein folding
  - ex. relativity computer
    - leave computer on earth, travel at speed of light for a while, come back and should be done
    - if you want exponential speedup, need to get exponentially close to speed of light (requires exponential energy)
  - ex. zeno's computer - run clock faster (ex;onentially more cooling = energy)

# quantum computing

- probability with minus signs
- *amplitudes* - used to calculate probabilites,   but can be negative / complex

![](assets/quantum/double_slit.png)

- applications
  - quantum simulation
  - also could factor integers in polynomial time (shor 1994)
  - scaling up is hard because of *decoherence*= interaction between cubits and outside world
  - error-correcting codes can make it so we can still work with some decoherence
- algorithms
  - paths that lead to wrong answer - quantum amplitudes cancel each other out
  - for right answer, quantum amplitudes in phase (all positive or all negative)
  - prime factorization is NP but not NP complete
  - unclear that quantum can solve all NP problems
  - *Grover's algorithm* - with quantum computers, something like you can only use sqrt of number of steps
  - *adiabatic optimization* - like quantum simulated annealing, maybe can solve NP-complete problems
- dwave - company made ~2000 cubit machine
  - don't maintain coherence well
  - algorithms for NP-complete problems may not work
  - hope: *quantum tunneling* can get past local maximum in polynomial time maybe
    - empircally unclear if this is true
- quantum supremacy - getting quantum speedup for something, maybe not something useful