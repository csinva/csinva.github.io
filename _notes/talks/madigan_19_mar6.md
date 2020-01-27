

# Towards honest inference from real-world healthcare data - david madigan (columbia)

- observational studies are wack - "real-world" buzzword
  - really just things you can buy: ehr, claims databases (diagnosis, procedures, etc.)
- dominant approach: propensity score matched analysis
  - people usually really focused on what 95% conf. interval contains
- these studies are neither **reliable nor reproducible**
  - most studies seem to be right around 0.05 cutoff
  - different databases give different results
  - things don't follow a protocol on things like missing values, ...
  - most authors still won't release code
  - different studies on same database still find different conclusions
  - analysis ignores selection bias, confounding, etc...

## a new approach

- goal: reproducible, systematized, open-source approach at scale
  - try everything
- negative controls
  - identify druges / outcomes "known" to have no causal association
  - get this ground truth by crawling literature / reports
  - negative control gives you null distribution (and empirical p-values)
- positive controls
  - inject signals into negative controls with knwon effect size
  - calibrated confidence intervals

## data

- OMOP common data model - works across 700 mil EHR records
  - a unified common format
- focused on population-level estimation, but also patient-level stuff
- clinical characterization - who uses what?

## examples

- negative controls: 68% have p < 0.05: unadjusted methods are dramatically wrong
  - when using propensity-based method, 16% have p < 0.05
  - using these controls, change the boundary for statistical significance to actually have 5% with p < 0.05
  - sometimes this produces a dramatic change in p-values of significance
- randomized trials have a number of strange idiosyncracies - hard to validate using these