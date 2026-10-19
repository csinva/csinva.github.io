# Detailed study of implicit bias in "Deep" Linear Learning

**Nati Srebro, TTIC**

- GD in matrix completion implicitly minimizes $$\vert \vert W\vert \vert _*$$
- GD in linear conv. net implicitly minimizes $$\vert \vert DFT(\beta)\vert \vert _p$$ - sparsity in freq. domain
- GD on infinite width relu net implicitly minimizes the maximum of a particular integral
  - suggests things go back to kernels
- under certain scales, trajectory of model behaves just like kernel (chizat & bach 18)
- just explicitly minimizing these norms doesn't necessarily gives same soln