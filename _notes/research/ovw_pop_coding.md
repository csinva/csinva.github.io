- saxena_19_pop_cunningham: "Towards the neural population doctrine"
  - correlated trial-to-trial variability
    - Ni et al. showed that the correlated variability in V4 neurons during attention and learning — processes that have inherently different timescales — robustly decreases 
    - ‘choice’ decoder built on neural activity in the first PC performs as well as one built on the full dataset, suggesting that the relationship of neural variability to behavior lies in a relatively small subspace of the state space.
  - decoding
    - more neurons only helps if neuron doesn't lie in span of previous neurons
  - encoding
    - can train dnn goal-driven or train dnn on the neural responses directly
  - testing
    - important to be able to test population structure directly



- population vector coding* - ex. neurons coded for direction sum to get final direction
- reduces uncertainty
- *correlation coding* - correlations betweeen spikes carries extra info
- *independent-spike coding* - each spike is independent of other spikes within the spike train
- *position coding* - want to represent a position
  - for grid cells, very efficient
- *sparse coding*
- hard when noise between neurons is correlated
- measures of information


## EDA

- plot neuron responses
- calc neuron covariances