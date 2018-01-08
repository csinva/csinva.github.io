---
layout: notes
section-type: notes
title: Math Basics
category: math
---

* TOC
{:toc}
# misc

- $\left( \frac{n}{k} \right) < \left( \frac{ne}{k} \right)^k$
- Stirling's formula: $ n! ~= (\frac{n}{e})^n $
  - corollary: log(n!) = 0(n log n)
  - gives us a bound on sorting
  - $\left( \frac{n}{e} \right)^n < n!$
- $(1-x)^N \leq e^{-Nx}$
- Poisson pmf approximates binomial when N large, p small



# functions

- *Gamma*: $\Gamma(n)=(n-1)!=\int_0^\infty x^{n-1}e^{-x}dx$
- *Zeta*: $\zeta(x) = \sum_1^\infty \frac{1}{x^2} $
- Sigmoid (logistic): $f(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{e^x+1}$
- Softmax: $f(x) = \frac{e^{x_i}}{\sum_i e^{x_i}}$

# stochastic processes

- Stochastic - random process evolving with time
- Markov: $P(X_t=x\|X_{t-1})=P(X_t=x\|X_{t-1}...X_1)$
- Martingale: $E[X_t]=X_{t-1}$ 

# abstract algebra

- Group: set of elements endowed with operation satisfying 4 properties:

1. closed 2. identity 3. associative 4. inverses

- Equivalence Relation;

1. reflexive 2. transitive 3. symmetric

# discrete math
- Goldbach's strong conjecture: Every even integer greater than 2 can be expressed as the sum of two primes (He considered one a prime).
- Goldbach's weak conjecture: All odd numbers greater than 7 are the sum of three primes.
- Set - An unordered collection of items without replication
- Proper subset - subset with cardinality less than the set
  - A U A = A			Idempotent law
- Disjoint: A and B = empty set
- Partition: mutually disjoint, union fills space
- powerset $\mathcal{P}$(A) = set of all subsets
- Converse: $q\to p$ (same as inverse: $-p \to -q$)
- $p_1 \to p_2 \iff - p_1 \lor p_2 $
- The greatest common divisor of two integers a and b is the largest integer d such that d $\|$ a and d $\|$ b
- Proof Techniques

1. Proof by Induction

2. Direct Proof

3. Proof by Contradiction - assume p $\land$ -q, show contradiction

4. Proof by Contrapositive - show -q $\to$ -p

# identities

- $e^{-2lnx}= \frac{1}{e^{2lnx}} = \frac{1}{e^{lnx}e^{lnx}} = \frac{1}{x^2}$
- $ln(xy) = ln(x)+ln(y)$
- $lnx * lny = ln(x^{lny})$
  - difference between log 10n and log 2n is always a constant (about 3.322)
- $e^{\mu it} = cos(\mu t)+ isin(\mu t)$
- Partial Fractions: $\frac{3x+11}{(x-3)(x+2)} = \frac{A}{x-3} + \frac{B}{x+2}$
- $(ax+b)^k = \frac{A_1}{ax+b}+\frac{A_2}{(ax+b)^2}+...$
- $(ax^2+bx+c)^k = \frac{A_1x+B_1}{ax^2+bx+c}+...$
- $cos(a\pm b) = cos(a)cos(b)\mp sin(a)sin(b)$
- $sin(a \pm b) = sin(a)cos(b) \pm sin(b)cos(a)$


