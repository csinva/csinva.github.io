# generalization bounds for neural networks via approximate description length

**amit daniely**

- natural thing to do is to consider generalization bounds based on weights / weight norms
  - bound frobenius norm of matrices by constant - should control the sample complexity
  - also bound spectral norm of weights - needed so output can't "explode"
- *approximate description length* - num bits required to *approximately* describe functions in a hypothesis class $\mathcal H$
  - e.g. if ADL = n $\implies$ sample complexity is $O(n/\epsilon^2)$
  - correct value with linear classes
  - behaves nicely with compositions
  - similar to covering number