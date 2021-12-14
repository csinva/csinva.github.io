# Machine Learning and Statistics for Climate Science

**karen mckinnon, andrew poppick - neurips 2021 tutorial**

- all systems are coupled (e.g. earth.nullschool.net)
- areas for ml <> climate science
  - heterogenous things - understand climate locally
  - causal inference
  - pairing ml with physical models
  - improve climate data via infilling, uncertainty estimation, etc.
- data
  - in situ measurements - direct, set up measurement (e.g. temperature) at location and record over time
    - land - global climate network temp. stations
    - ocean - harder to measure, best measured along shipping roats
      - argo program (~2005) - started recording using autonomous floats
  - satellite data - challenge: can't see through clouds, sometimes doesn't properly quantify what we want
    - ex. land-surface temperature
    - ex. SMAP satellite - soil moisture + ocean salinity
    - doesn't go far back (e.g. starts ~1979)
  - gridded data
    - noaa oisst v2 - satellite + in situe daily ocean
    - berkely earth surface tempretaures - interpolated in situ measurements, monthly
  - reanalyses - combination of many data sources
    - e.g. ECMWF - physical model for weather which assimilates data
  - proxy records / archives - ex. tree rings
    - long time scales
  - climate models / earth system models
    - ex. carbonbrief.org
    - divide earth into grid cells (can be too coarse)
      - interact using known equations
      - very computationally expensive
    - easy to do experiments
    - lots of noise, especially on small scales
  - challenges
    - inhomogeneities - observing platforms change over time