# nonlinear ica

**aapo hyvarinen, University of Helsinki**

- nonlinear ica fundamentally ill-defined: not identifiable
  - we can learn different nonlinear functions of non-Gaussian sources that result in the observationsf
- soln: impose structure
  - ex. temporal structure in time-series
  - ex. use auxiliary variable such as audio for video
- linear ICA - latent sources $s_j$ are independent and non-Gaussian
  - these are linearly mixed to produced observations
  - identifiable - given only x we can recover sources and coefficients
  - best-case against PCA: separates uniformly distributed sources
  - things like PCA are not identifiable bc they cannot separate orthogonal rotations
- self-supervised learning - only given x use supervised learning on someting dervied from x



## temporal structure helps in nonlinear ica

- time-contrastive learning (hyvarinen et al. 2016)
  - divide x into even segments then train mlp to tell which segment a single data point comes from
  - NN learns nonstationarity (difference between segments)
  - this can provide identifiability
  - intuitively this imposes independence at every segment, which is like introducing more constraints
- permutation-contrastive learning (hyvarinen and morioka 2017)
  - take short time windows $y$
  - create randomly time-permuted data $y^*$
  - train NN to discriminate $y$ from $y*$
  - this performs nonlinear ica for temporally dependent components
- follow-up work
  - combines these 2 methods (nonstationary innovations, morioka & hyvarinene, arxiv)
  - also segmentation can be learned (combine tcl with HMM, halva & hyarinen, 2020)



## general framework: deep latent variable models

- VAE kind of looks like nonlinear ICA except for noise term
  - however, by Gaussianity, any orthogonal rotation of the sources is equivalent, so VAE is not identifiable
  - more like PCA rathern than ICA
- conditionaing dvlm's by another variable (hyvarinen et al. 2019)
  - observe $u$ which modulates components $p(s_i|u)$ (e.g. $u$ is audio for video)
  - $s_i$ is conditionally independent given $u$
  - now, it is identifiable e.g. iVAE (khemakhem et al. 2020)
- generalized to dependent components using energy-baesd modeling (khemakhem et al. arxiv)