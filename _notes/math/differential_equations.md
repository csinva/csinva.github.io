---
layout: notes
section-type: notes
title: Differential Equations
category: math
---

# Differential Equations
Separable: Separate and Integrate
FOLDE: y' + p(x)y = g(x)
IF: $e^{\int{p(x)}dx}$

Exact: Mdx+Ndy = 0 $M_y=N_x$ 
Integrate Mdx or Ndy, make sure all terms are present

Constant Coefficients: 
Plug in $e^{rt}$, solve characteristic polynomial
repeated root solutions: $e^{rt},re^{rt}$
complex root solutions: $r=a\pm bi, y=c_1e^{at} cos(bt)+c_2e^{at} sin(bt)$

SOLDE (non-constant): 
py''+qy'+ry=0

Reduction of Order: Know one solution, can find other

Undetermined Coefficients (doesn't have to be homogenous): solve homogenous first, then plug in form of solution with variable coefficients, solve polynomial to get the coefficients

Variation of Parameters: start with homogenous solutions $y_1,y_2$
$Y_p=-y_1\int \frac{y_2g}{W(y_1,y_2)}dt+y_2\int \frac{y_1g}{W(y_1,y_2)}dt$

Laplace Transforms - for anything, best when g is noncontinuous

$\mathcal{L}(f(t))=F(t)=\int_0^\infty e^{-st}f(t)dt$

Series Solutions: More difficult

Wronskian: $W(y_1 ,y_2)=y_1y _2' -y_2 y_1'$
W = 0 $\implies$ solns linearly dependent

Abel's Thm: y''+py'+q=0 $\implies W=ce^{\int pdt}$