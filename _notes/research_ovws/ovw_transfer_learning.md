---
layout: notes
title: transfer learning
category: research
---



See also notes on causal inference for some close connections.



- [Domain-Adversarial Training of Neural Networks](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf) (ganin et al. 16) - want repr. to be invariant to domain label
  - ![Screen Shot 2020-11-10 at 12.05.12 PM](../assets/domain_adv_training.png)
  - exact same algorithm is used to [learn fair representations](https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf), except domain label is replaced with sensitive attribute
- domain adaptation, given multiple training groups
  - [group](http://papers.neurips.cc/paper/3019-mixture-regression-for-covariate-shift.pdf) [distributionally robust](https://arxiv.org/abs/1611.02041) [optimization](https://arxiv.org/abs/1911.08731)
  - [domain](https://papers.nips.cc/paper/4312-generalizing-from-several-related-classification-tasks-to-a-new-unlabeled-sample) [generalization](https://arxiv.org/abs/2007.01434)
- domain adaptation using source/target, given all at once
  - [importance weighting](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.370.4921&rep=rep1&type=pdf)
  - learning [invariant represetntations](https://arxiv.org/abs/1702.05464)
- test-time adaptation
  - [batch normalization](https://arxiv.org/abs/1603.04779)
  - [label shift estimation](https://arxiv.org/abs/1802.03916)
  - [rotation prediction](https://arxiv.org/abs/1909.13231) (sun et al. 2020)
  - [entropy minimization](https://arxiv.org/abs/2006.10726)
- [adaptive risk minimization](https://arxiv.org/abs/2007.02931) - combines groups at training time + batches at test-time
  - *meta-train* the model using simulated distribution shifts, which is enabled by the training groups, such that it exhibits strong *post-adaptation* performance on each shift
- [Domain adaptation under structural causal models](https://arxiv.org/abs/2010.15764) (chen & buhlmann, 2020)
  - make clearer assumptions for domain adaptation to work
  - introduce CIRM, which works better when both covariates and labels are perturbed in target data

# adv attacks

- [Adversarial Attacks and Defenses in Images, Graphs and Text: A Review](https://arxiv.org/abs/1909.08072) (xu et al. 2019) 
- attacks
  - [Barrage of Random Transforms for Adversarially Robust Defense](http://openaccess.thecvf.com/content_CVPR_2019/papers/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.pdf) (raff et al. 2019) 
  - [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/abs/1511.04599) (Moosavi-Dezfooli et. al 2016)
- defenses
  - a possible defense against adversarial attacks is to solve the anticausal classification problem by modeling the causal generative direction, a method which in vision is referred to as *analysis by synthesis* ([Schott et al., 2019](https://arxiv.org/abs/1805.09190))
- robustness vs accuracy
  - [robustness may be at odds with accuracy](https://openreview.net/pdf?id=SyxAb30cY7) (tsipiras...madry, 2019)
  - [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/abs/1901.08573) (Zhang, Jordan, et. al. 2019)
- adversarial examples
  - [Decision Boundary Analysis of Adversarial Examples ](https://pdfs.semanticscholar.org/08c5/88465b7d801ad912ef3e9107fa511ea0e403.pdf)(He, Li, & Song 2019)
  - [Natural Adversarial Examples](https://arxiv.org/abs/1907.07174) (Hendrycks, Zhao, Basart, Steinhardt, & Song 2020)
  - [Image-Net-Trained CNNs Are Biased Towards Texture](https://openreview.net/pdf?id=Bygh9j09KX) (Geirhos et al. 2019)
- transferability
  - [Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples](https://arxiv.org/abs/1605.07277) (papernot, mcdaniel, & goodfellow, 2016)
  - [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/pdf/1705.07204.pdf) (tramer et al. 2018)
  - [Improving Adversarial Robustness via Promoting Ensemble Diversity](https://arxiv.org/pdf/1901.08846.pdf) (pang et al. 2019)
    - encourage diversity in non-maximal predictions
- ranking
  - [Automatically Discovering and Learning New Visual Categories with Ranking Statistics](https://arxiv.org/pdf/2002.05714.pdf)