---
layout: notes
title: generalization
category: research
---

**some notes on generalization bounds in machine learning with an emphasis on stability**

{:toc}

- Risk: $R\left(A_{S}\right)=\mathbb{E}_{(X, Y) \sim P} \ell\left(A_{S}(X), Y\right)$
- Empirical risk: $R_{\mathrm{emp}}\left(A_{S}\right)=\frac{1}{n} \sum_{i=1}^{n} \ell\left(A_{S}\left(X_{i}\right), Y_{i}\right)$
- Generalization error: $R\left(A_{S}\right)-R_{\mathrm{emp}}\left(A_{S}\right)$
- [Sharper Bounds for Uniformly Stable Algorithms](http://proceedings.mlr.press/v125/bousquet20b.html)
  - uniformly stable with prameter $\gamma$ if $\left|\ell\left(A_{S}(x), y\right)-\ell\left(A_{S^{i}}(x), y\right)\right| \leq \gamma$
    - where $A_{S^i}$ can alter one point in the training data
  - stability approach can provide bounds even when empirical risk is 0 (e.g. 1-nearest-neighbor, devroye & wagner 1979)
  - $n\underbrace{\left(R\left(A_{S}\right)-R_{\mathrm{emp}}\left(A_{S}\right)\right)}_{\text{generalization error}} \lesssim(n \sqrt{n} \gamma+L \sqrt{n}) \sqrt{\log \left(\frac{1}{\delta}\right)}$ (bousegeuet & elisseef, 2002)

