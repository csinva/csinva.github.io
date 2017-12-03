---
layout: notes
section-type: notes
title: Math Basics
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


# Misc
$\Gamma(n)=(n-1)!=\int_0^\infty x^{n-1}e^{-x}dx$

$\zeta(x) = \sum_1^\infty \frac{1}{x^2} $
\end{multicols}


# stochastic processes
Stochastic - random process evolving with time

Markov: $P(X_t=x\|X_{t-1})=P(X_t=x\|X_{t-1}...X_1)$

Martingale: $E[X_t]=X_{t-1}$ 


# abstract algebra
Group: set of elements endowed with operation satisfying 4 properties:
1. closed 2. identity 3. associative 4. inverses

Equivalence Relation;
1. reflexive 2. transitive 3. symmetric

# discrete math
Goldbach's strong conjecture: Every even integer greater than 2 can be expressed as the sum of two primes (He considered one a prime).

Goldbach's weak conjecture: All odd numbers greater than 7 are the sum of three primes.

Set - An unordered collection of items without replication

Proper subset - subset with cardinality less than the set

A U A = A				Idempotent law

Disjoint: A and B = empty set

Partition: mutually disjoint, union fills space

powerset $\mathcal{P}$(A) = set of all subsets

Converse: $q\ra p$ (same as inverse: $-p \ra -q$)

$p_1 \ra p_2 \iff - p_1 \lor p_2 $

The greatest common divisor of two integers a and b is the largest integer d such that d $\|$ a and d $\|$ b

Proof Techniques

1. Proof by Induction

2. Direct Proof

3. Proof by Contradiction - assume p $\land$ -q, show contradiction

4. Proof by Contrapositive - show -q $\ra$ -p

# identities
$e^{-2lnx}= \frac{1}{e^{2lnx}} = \frac{1}{e^{lnx}e^{lnx}} = \frac{1}{x^2}$

$ln(xy) = ln(x)+ln(y)$

$lnx * lny = ln(x^{lny})$

$e^{\mu it} = cos(\mu t)+ isin(\mu t)$

Partial Fractions
$\frac{3x+11}{(x-3)(x+2)} = \frac{A}{x-3} + \frac{B}{x+2}$

$(ax+b)^k = \frac{A_1}{ax+b}+\frac{A_2}{(ax+b)^2}+...$

$(ax^2+bx+c)^k = \frac{A_1x+B_1}{ax^2+bx+c}+...$

$cos(a\pm b) = cos(a)cos(b)\mp sin(a)sin(b)$

$sin(a \pm b) = sin(a)cos(b) \pm sin(b)cos(a)$

