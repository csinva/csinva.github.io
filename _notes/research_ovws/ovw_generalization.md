---
layout: notes
title: Generalization
category: research
subtitle: Some notes on generalization bounds in machine learning with an emphasis on stability
---

{:toc}

# basics
- Risk: $R\left(A_{S}\right)=\mathbb{E}_{(X, Y) \sim P} \ell\left(A_{S}(X), Y\right)$
- Empirical risk: $R_{\mathrm{emp}}\left(A_{S}\right)=\frac{1}{n} \sum_{i=1}^{n} \ell\left(A_{S}\left(X_{i}\right), Y_{i}\right)$
- Generalization error: $R\left(A_{S}\right)-R_{\mathrm{emp}}\left(A_{S}\right)$

# uniform stability

- [Sharper Bounds for Uniformly Stable Algorithms](http://proceedings.mlr.press/v125/bousquet20b.html)
  - uniformly stable with prameter $\gamma$ if $\left|\ell\left(A_{S}(x), y\right)-\ell\left(A_{S^{i}}(x), y\right)\right| \leq \gamma$
    - where $A_{S^i}$ can alter one point in the training data
  - stability approach can provide bounds even when empirical risk is 0 (e.g. 1-nearest-neighbor, devroye & wagner 1979)
  - $n\underbrace{\left(R\left(A_{S}\right)-R_{\mathrm{emp}}\left(A_{S}\right)\right)}_{\text{generalization error}} \lesssim(n \sqrt{n} \gamma+L \sqrt{n}) \sqrt{\log \left(\frac{1}{\delta}\right)}$ (bousegeuet & elisseef, 2002)



# trees

- [A cautionary tale on fitting decision trees to data from additive models: generalization lower bounds](https://arxiv.org/abs/2110.09626) (tan, agarwal, & yu, 2021)
  - (generalized) additive models: $y = f(X_1) + f(X_2) + ... f(X_p)$ (as opposed to linear models)
  - main result
    - lower bound: can't peform better than $\Omega (n^{-\frac{2}{s+2}})$ in $\ell_2$ risk (as defined above)
      - best tree (ALA tree) can't do better than this bound (e.g. risk $\geq \frac{1}{n^{s+2}}$ times some stuff)
  - generative assumption simplifies calculations by avoiding some of the complex dependencies be- tween splits that may accumulate during recursive splitting
  - ALA tree
    - learns axis-aligned splits
    - makes predictions as leaf averages
    - honest trees: val split is used to estimate leaf averages
