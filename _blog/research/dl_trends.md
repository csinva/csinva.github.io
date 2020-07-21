---
layout: notes
title: recent cool results in deep learning
category: blog
---

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQR_qH6bY6iJyxJam0u3l6_jaurDtvpyr0L-m43F0YOQbbOdnsQYormbfri92aSdbdVK9bQ8ySneaqe/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

**some notes accompanying the above presentation**

- supervised vision results
  - [DenseNet](https://arxiv.org/abs/1608.06993) (‘17)
  - [efficientnet](https://arxiv.org/pdf/1905.11946.pdf)
  - new imagenet datasets
    - e.g. imagenet 2, imagenet-c, imagenet-a
- unsupervised learning
  - gans (stylegan 1/2, mode collapse)
  - vaes (alae, [NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898))
- Improvements in other domains
  - [openai jukebox](https://arxiv.org/abs/2005.00341) (VQ-VAE)
- semi-supervised learning
  - transformers ([gpt3](https://arxiv.org/abs/2005.14165), [bert](https://arxiv.org/abs/1810.04805), [image gpt](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V1_ICML.pdf))
    - [gpt 3 tweets](https://twitter.com/edleonklinger/status/1284251419172909057)
  - [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192) (alyosha's group, 2015)
- more leveraging unlabeled data
  - images ([simclr](https://arxiv.org/abs/2002.05709), [simclr2](https://arxiv.org/pdf/2006.10029.pdf))
  - [Self-training with Noisy Student improves ImageNet classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf) 
- related applications
  - gesture from voice
  - making videos
  - [cyclegan](https://arxiv.org/abs/1703.10593) (2018)
- games (e.g. [here](https://en.wikipedia.org/wiki/Progress_in_artificial_intelligence#cite_note-time_top_10-9))
  - 1992 - [TD-GAMMON](https://cling.csd.uwo.ca/cs346a/extra/tdgammon.pdf)  uses RL (in 3-layer FNN)  to be almost champion-level
  - 1996 - deepblue beats gary kasparov (alpha-beta search) - not DL
  - 2016 - [alphago](https://ai.googleblog.com/2016/01/alphago-mastering-ancient-game-of-go.html) beats Lee Sedol - combines tree search with DL
  - 2017 - [alphago zero](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ) - works without human knowledge
  - 2017 - [DeepStack](https://science.sciencemag.org/content/356/6337/508) / [Libratus](https://science.sciencemag.org/content/359/6374/418)
  - 2018 - [alpha zero](https://science.sciencemag.org/content/362/6419/1140.full?ijkey=XGd77kI6W4rSc&keytype=ref&siteid=sci) - same rl masters chess, shogi, Go
  - 2019 - [alphastar](https://www.nature.com/articles/s41586-019-1724-z.epdf?author_access_token=lZH3nqPYtWJXfDA10W0CNNRgN0jAjWel9jnR3ZoTv0PSZcPzJFGNAZhOlk4deBCKzKm70KfinloafEF1bCCXL6IIHHgKaDkaTkBcTEv7aT-wqDoG1VeO9-wO3GEoAMF9bAOt7mJ0RWQnRVMbyfgH9A%3D%3D) grandmaster level at starcraft II
- reinforcement learning
  - [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/pdf/1705.05363.pdf) (efros & darrell, 2018)
  - [ensembling is all you need](https://arxiv.org/abs/2007.04938) (abbeel group, 2020)
  - [image augmentation is all you need](https://arxiv.org/abs/2004.13649) (fergus group, 2020) + [reinforcement learning with augmented data](https://arxiv.org/abs/2004.14990) (abbeel group, 2020)
  - [test-time training](https://arxiv.org/pdf/2007.04309.pdf)
- succesful applications
  - alphafold
  - retinopathy...
- trustworthy ml
  - uncertainty
  - LIME/SHAP
- emerging concerns
  - pulse
  - biased models
  - biased data
    - [large image datasets issues](https://openreview.net/pdf?id=s-e2zaAlG3I)
  - environmental impacts
  - [shifting power](https://www.nature.com/articles/d41586-020-02003-2)  (kalluri 2020) 
    - e.g. facial recognition
- fake news
  
- interesting stuff
  - [neural ODEs](https://arxiv.org/abs/1806.07366) (2019)
  - [capsule networks](https://arxiv.org/abs/1710.09829) (2017)
  - graph neural networks
- cool empirical investigations
  - lottery ticket hypothesis
  - bagnet