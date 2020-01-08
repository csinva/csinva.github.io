---
layout: notes
title: real analysis
category: math
---

* TOC
{:toc}

# ch 1 - the real numbers
- there is no rational number whose square is 2 <button data-toggle="collapse" data-target="#111" >+</button><div class="collapse" id="111">
proof by contradiction </div>
- *contrapositive*: $$-q \to -p$$ - logically equivalent
- *triangle inequality*: $\|a+b\| \leq \|a\| + \|b\|$<button data-toggle="collapse" data-target="#121" >+</button><div class="collapse" id="121">
often use \|a-b\| = \|(a-c)+(c-b)\|</div>
- *axiom of completeness* - every nonempty set $A \subseteq \mathbb{R}$ that is bounded above has a least upper bound
	- doesn't work for $\mathbb{Q}$
- *supremum* = supA = least upper bound (similarly, *infimum*)
	1. supA is an upper bound of A
	2. if $s \in \mathbb{R}$ is another u.b. then $s \geq supA$
		- can be restated as $\forall \epsilon > 0, \exists a \in A$ $s-\epsilon < a$
- *nested interval property* - for each $n \in N$, assume we are given a closed interval $I_n = [a_n,b_n]=\{ x \in \mathbb{R} : a_n \leq x \leq b_n \}$  Assume also that each $I_n$ contains $I_{n+1}$.  Then, the resulting nested sequence of nonempty closed intervals $I_1 \supseteq I_2 \supseteq ...$ has a nonempty intersection <button data-toggle="collapse" data-target="#141" >+</button><div class="collapse" id="141">
use AoC with x = sup{$a_n: n \in \mathbb{N}$} in the intersection of all sets</div>
- *archimedean property*<button data-toggle="collapse" data-target="#142" >+</button>
	1. $\mathbb{N}$ is unbounded above (sup $\mathbb{N}=\infty$)
	2. $\forall x \in \mathbb{R}, x>0, \exists n \in \mathbb{N}, 0 < \frac{1}{n} < x$
<div class="collapse" id="142">contradiction with AoC</div>
- $\mathbb{Q}$ is dense in $\mathbb{R}$ - for every $a,b \in \mathbb{R}, a<b$, $\exists r \in \mathbb{Q}$ s.t. $a<r<b$
	- pf: want $a < \frac{m}{n} < b$
		- by Archimedean property, want $\frac{1}{n} < b-a$
	- corollary: the irrationals are dense in $\mathbb{R}$
- there exists a real number $r \in \mathbb{R}$ satisfying $r^2 = 2$
	- pf: let r = $sup \{ t \in \mathbb{R} : t^2 < 2 \}$.  disprove $r^2<2, r^2>2$ by considering $r+\frac{1}{n},r-\frac{1}{n}$
- A ~ B if there exists f:A->B that is 1-1 and onto
- A is *finite* - there exists n $\in \mathbb{N}$ s.t. $\mathbb{N}_n$~A
- *countable* =  $\mathbb{N}$~A.  
	- uncountable - inifinite set that isn't countable
	- Q is countable
		- pf: Let $A_n = \{ \pm \frac{p}{q}:$ where p,q $\in \mathbb{N}$ are in lowest terms with p+q=n}
	- R is uncountable
		- pf: Assume we can enumerate $\mathbb{R}$  Use NIP to exclude one point from $\mathbb{R}$ each time.  The intersection is still nonempty, so we didn't succesfully enumerate $\mathbb{R}$
	- $\frac{x}{x^2-1}$ maps (0,1) $\to \mathbb{R}$
	- countable union of countable sets is countable
- if $A \subseteq B$ and B countable, then A is either countable or finite
- if $A_n$ is a countable set for each $n \in \mathbb{N}$, then their union is countable
- the open interval (0,1) = $\{ x \in \mathbb{R} : 0 < x < 1 \}$ is uncountable
	- pf: diagonalization - assume there exists a function from (0,1) to $\mathbb{R}$.  List the decimal expansions of these as rows of a matrix.  Complement of diagonal does not exist.
- *cantor's thm* - Given any set A, there does not exist a function f:$A \to P(A)$ that is onto
	- P(A) is the set of all subsets of A

# ch 2 - sequences and series
- a sequence $(a_n)$ *converges* to a real number if $\forall \epsilon > 0, \exists N \in \mathbb{N}$ such that $\forall n\geq N, \|a_n-a\| < \epsilon$
	
	- otherwise it *diverges*
- if a limit exists, it is unique
- a sequence $(x_n)$ is *bounded* if there exists a number M > 0 such that $\|x_n\|\leq M \forall n \in \mathbb{N}$
	
	- every convergent sequence is bounded
- *algebraic limit thm* - let lim $a_n = a$ and lim $b_n$ = b.  Then
	1. lim($ca_n$) = ca
	2. lim($a_n+b_n$) = a+b
	3. lim($a_n b_n$) = ab
	4. lim($a_n/b_n$) = a/b, provided b $\neq$ 0
	- pf 3: use triangle inequality, $\|a_nb_n-ab\|=\|a_nb_n-ab_n+ab_n-ab\|=...=\|b_n\|\|a_n-a\|+\|a\|\|b_n-b\|$
	- pf 4: show $(b_n) \to b$ implies $(\frac{1}{b_n}) \to \frac{1}{b}$
- *order limit thm* - Assume lim $a_n = a$ and lim $b_n$ = b.
	1. If $a_n \geq 0$ $\forall n \in \mathbb{N}$, then $a \geq 0$
	2. If $a_n \leq b_n$ $\forall n \in \mathbb{N}$, then $a \leq b$
	3. If $\exists c \in \mathbb{R}$ for which $c \leq b_n$ $\forall n \in \mathbb{N}$, then $c \leq b$
	- pf 1: by contradiction
- *monotone* - increasing or decreasing (not strictly)
- *monotone convergence thm* - if a sequence is monotone and bounded, then it converges
- *convergence of a series*
	- define $s_m=a_1+a_2+...+a_m$
	- $\sum_{n=1}^\infty a_n$ converges to A $\iff (s_m)$ converges to A
- *cauchy condensation test* - suppose $a_n$ is decreasing and satisfies $a_n \geq 0$ for all $n \in \mathbb{N}$.  Then, the series $\sum_{n=1}^\infty a_n$ converges iff the series $\sum_{n=1}^\infty 2^na_{2^n}$ converges
	- *p-series* $\sum_{n=1}^\infty 1/n^p$ converges iff p > 1
	
## 2.5
- let $(a_n)$ be a sequence and $n_1<n_2<...$ be an increasing sequence of natural numbers.  Then $(a_{n_1},a_{n_2},...)$ is a *subsequence* of $(a_n)$
- subsequences of a convergent sequence converge to the same limit as the original sequence
	- can be used as divergence criterion
- *bolzano-weierstrass thm* - every bounded sequence contains a convergent subsequence
	- pf: use NIP, keep splitting interval into two

## 2.6
- $(a_n)$ is a *cauchy sequence* if $\forall \epsilon > 0, \exists N \in \mathbb{N}$ such that $\forall m,n\geq N, \|a_n-a_m\| < \epsilon$
- *cauchy criterion* - a sequence converges $\iff$ it is a cauchy sequence
	- cauchy sequences are bounded
- overview: AoC $\iff$ NIP $\iff$ MCT $\iff$ BW $\iff$ CC

## 2.7
- *algebraic limit thm* - let $\sum_{n=1}^\infty a_n$ = A, $\sum_{n=1}^\infty b_n$ = B
	1. $\sum_{n=1}^\infty ca_n$ = cA
	2. $\sum_{n=1}^\infty a_n+b_n$ = A+B
1. *cauchy criterion for series* - series converges $\iff$ $(s_m)$ is a cauchy sequence
- if the series $\sum_{n=1}^\infty a_n$ converges then lim $a_n=0$
2. *comparison test*
3. *geometric series* - 	$\sum_{n=0}^\infty a r^n = \frac{a}{1-r}$
	
	- $s_m = a+ar+...+ar^{m-1} = \frac{a(1-r^m)}{1-r}$
4. *absolute convergence test*
5. *alternating series test*
	1. decreasing
	2. lim $a_n$ = 0
	- then, $\sum_{n=1}^\infty (-1)^{n+1} a_n$ converges
- rearrangements: there exists one-to-one correspondence
- if a series converges absolutely, any rearrangement converges to same limit	

# ch 3 - basic topology of R

## 3.1 cantor set
- C has small length, but its cardinality is uncountable
- discussion of dimensions, doubling sizes leads to 2^dimension sizes
	- Cantor set is about dimension .631
	
## 3.2 open/closed sets
- A set O $\subseteq \mathbb{R}$ is *open* if for all points a $\in$ O there exists an $\epsilon$-neighborhood $V_{\epsilon}(a) \subseteq O$
	-  $V_{\epsilon}(a)=\{ x \in R : \|x-a\| < \epsilon$}
1. the union of an arbitrary collection of open sets is open
2. the intersection of a finite collection of open sets is open
- a point x is a *limit point* of a set A if every $\epsilon$-neighborhood $V_{\epsilon}(x)$ of x intersects the set A at some point other than x
- a point x is a limit point of a set A if and only if x = lim $a_n$ for some sequence ($a_n$) contained in A satisfying $a_n \neq x$ for all n $\in$ N
- *isolated point* - not a limit point
- set $F \subseteq \mathbb{R}$ *closed* - contains all limit points
	- *closed* iff every Cauchy sequence contained in F has a limit that is also an element of F
- density of Q in R - for every $y \in \mathbb{R}$, there exists a sequence of rational numbers that converges to y
- *closure* - set with its limit points
	- closure $\bar{A}$ is smallest closed set containing A
- iff set open, complement is closed
	- R and $\emptyset$ are both open and closed
1. the union of a finite collection of closed sets is closed
2. the intersection of an arbitrary collection of closed sets is closed

## 3.3 
- a set K $\subseteq \mathbb{R}$ is *compact* if every sequence in K has a subsequence that converges to a limit that is also in K
- *Nested Compact Set Property* - intersection of nested sequence of nonempty compact sets is not empty
- let A $\subseteq \mathbb{R}$.  *open cover* for A is a (possibly infinite) collection of  open sets whose union contains the set A.  
- given an open cover for A, a *finite subcover* is a finite sub-collection of open sets from the original open cover whose union still manages to completely contain A
- *Heine-Borel thm* - let K $\subseteq \mathbb{R}$.  All of the following are equivalent
	1. K is compact
	2. K is closed and bounded
	3. every open cover for K has a finite subcover
	
# ch 4	- functional limits and continuity

## 4.1 
- dirichlet function: 1 if r $\in \mathbb{Q}$ 0 otherwise	
	
## 4.2 functional limits
- def 1. Let f:$A \to R$, and let c be a limit point of the domain A.  We say that *$lim_{x \to c} f(x) = L$* provided that for all $\epsilon$ > 0, there exists a $\delta$ > 0 s.t. whenever 0 < \|x-c\| < $\delta$ (and x $\in$ A) it follows that \|f(x)-L\|< $\epsilon$
- def 2. Let f:$A \to R$, and let c be a limit point of the domain A. We say that $lim_{x \to c} f(x) = L$ provided that for every $\epsilon$-neighborhood $V_{\epsilon}(L)$ of L, there exists a $\delta$-neighborhood $V_{\delta}($c) around c with the property that for all x $\in V_{\delta}($c) different from c (with x $\in$ A) it follows that f(x) $\in V_{\epsilon}(L)$.
- *sequential criterion for functional limits* - Given function f:$A \to R$ and a limit point c of A, the following 2 statements are equivalent:
	1. $lim_{x \to c} f(x) = L$
	2. for all sequences $(x_n) \subseteq$ A satisfying $x_n \neq$ c and $(x_n) \to c$, it follows that $f(x_n) \to L$.
- *algebraic limit thm for functional limits*
- *divergence criterion for functional limits*

## 4.3 continuous functions
- a function f:$A \to R$ is *continuous at a point* c $\in$ A if, for all $\epsilon$>0, there exists a $\delta$>0 such that whenever \|x-c\|<$\delta$ (and x$\in$ A) it follows that $\|f(x)-f( c)\|<\epsilon$.  F is *continous* if it is continuous at every point in the domain A
- characterizations of continuouty
- criterion for discontinuity
- algebraic continuity theorem
- if f is continuous at c and g is continous at f( c) then g $\circ$ f is continuous at c

## 4.4 continuous functions on compact sets
- *preservation of compact sets* - if f continuous and K compact, then f(K) is compact as well
- *extreme value theorem* - if f if continuous on a compact set K, then f attains a maximum and minimum value.  In other words, there exist $x_0,x_1 \in K$ such that $f(x_0) \leq f(x) \leq f(x_1)$ for all x $\in$ K
- f is *uniformly continuous on A* if for every $\epsilon$>0, there exists a $\delta$>0 such that for all x,y $\in$ A, $\|x-y\| < \delta \implies \|f(x)-f(y)\| < \epsilon$
	- a function f fails to be uniformly continuous on A iff there exists a particular $\epsilon_o$ > 0 and two sequences $(x_n),(y_n)$ in A sastisfying $\|x_n - y_n\| \to 0$ but $\|f(x_n)-f(y_n)\| \geq \epsilon_o$
- a function that is continuous on a compact set K is *uniformly continuous on K*

## 4.5 intermediate value theorem
- *intermediate value theorem* - Let f:[a,b]$ \to R$ be continuous.  If L is a real number satisfying f(a) < L < f(b) or f(a) > L > f(b), then there exists a point c $\in (a,b)$ where f( c) = L
- a function f has the *intermediate value property* on an inverval [a,b] if for all x < y in [a,n] and all L between f(x) and f(y), it is always possible to find a point c $\in (x,y)$ where f( c)=L.

# ch 5 - the derivative

## 5.2 derivatives and the intermediate value property
- let g: A -> R be a function defined on an interval A.  Given c $\in$ A, the *derivative of g at c* is defined by g'( c) = $\lim_{x \to c} \frac{g(x) - g( c)}{x-c}$, provided this limit exists.  Then g is differentiable at c.  If g' exists for all points in A, we say g is *differentiable* on A
- identity: $x^n-c^n = (x-c)(x^{n-1}+cx^{n-2}+c^2x^{n-3}+...+c^{n-1}$)
- differentiable $\implies$ continuous
- *algebraic differentiability theorem*
	1. adding
	2. scalar multiplying
	3. product rule
	4. quotient rule
- *chain rule*: let f:A-> R and g:B->R satisfy f(A)$\subseteq$ B so that the composition g $\circ$ f is defined.  If f is differentiable at c in A and g differentiable at f( c) in B, then g $\circ$ f is differnetiable at c with (g$\circ$f)'( c)=g'(f( c))*f'( c)
- *interior extremum thm* - let f be differentiable on an open interval (a,b).  If f attains a maximum or minimum value at some point c $\in$ (a,b), then f'( c) = 0.  
- *Darboux's thm* - if f is differentiable on an interval [a,b], and a satisfies f'(a) < $\alpha$ < f'(b)  (or f'(a) > $\alpha$ > f'(b)), then there exists a point c $\in (a,b)$ where f'( c) = $\alpha$
	- derivative satisfies intermediate value property

## 5.3 mean value theorems
- *mean value theorem* - if f:[a,b] -> R is continuous on [a,b] and differentiable on (a,b), then there exists a point c $\in$ (a,b) where $f'( c) = \frac{f(b)-f(a)}{b-a}$
	- *Rolle's thm* - f(a)=f(b) -> f'( c)=0
	- if f'(x) = 0 for all x in A, then f(x) = k for some constant k
- if f and g are differentiable functions on an interval A and satisfy f'(x) = g'(x) for all x $\in$ A, then f(x) = g(x) + k for some constant k
- *generalized mean value theorem* - if f and g are continuous on the closed interval [a,b] and differentiable on the open interval (a,b), then there exists a point c $\in (a,b)$ where \|f(b)-f(a)\|g'( c) = \|g(b)-g(a)\|f'( c).  If g' is never 0 on (a,b), then can be restated $\frac{f'( c)}{g'( c)} = \frac{f(b)-f(a)}{g(b)-g(a)}$
- given g: A -> R and a limit point c of A, we say that *$lim_{x \to c} g(x) = \infty$* if, for every M > 0, there exists a $\delta$> 0 such that whenever 0 < \|x-c\| < $\delta$ it follows that g(x) ≥ M
- *L'Hospital's Rule: 0/0* - let f and g be continuous on an interval containing a, and assume f and g are differentiable on this interval with the possible exception of the point a.  If f(a) = g(a) = 0 and g'(x) ≠ 0 for all x ≠ a, then $lim_{x \to a} \frac{f'(x)}{g'(x)} = L \implies lim_{x \to a} \frac{f'(x)}{g'(x)} = L$
- *L'Hospital's Rule: $\infty / \infty$* - assume f and g are differentiable on (a,b)  and g'(x) ≠ 0 for all x in (a,b).  If $lim_{x \to a} g(x) = \infty $, then $lim_{x \to a} \frac{f'(x)}{g'(x)} = L \implies lim_{x \to a} \frac{f'(x)}{g'(x)} = L$

# ch 6 - sequences and series of function

## 6.2 uniform convergence of a sequence of functions
- for each n $\in \mathbb{N}$ let $f_n$ be a function defined on a set A$\subseteq R$.  The sequence ($f_n$) of functions *converges pointwise* on A to a function f if, for all x in A, the sequence of real numbers $f_n(x)$ converges to f(x)
- let ($f_n$) be a sequence of functions defined on a set A$\subseteq$R.  Then ($f_n$) *converges unformly* on A to a limit function f defined on A if, for every $\epsilon$>0, there exists an N in $\mathbb{N}$ such that $\forall n ≥N,  x \in A , \|f_n(x)-f(x)\|<\epsilon$
	- *Cauchy Criterion* for uniform convergence - a sequence of functions $(f_n)$ defined on a set A $\subseteq$ R  converges uniformly on A iff $\forall \epsilon > 0 \exists N \in \mathbb{N}$ s.t. whenever m,n ≥N and x in A, $\|f_n(x)-f_m(x)\|<\epsilon$
- *continuous limit thm* - Let ($f_n$) be a sequence of functions defined on A that converges uniformly on A to a function f.  If each $f_n$ is continuous at c in A, then f is continuous at c

## 6.3 uniform convergence and differentiation
- *differentiable limit theorem* - let $f_n \to f$ pointwise on the closed interval [a,b], and assume that each $f_n$ is differentiable.  If $(f'_n)$ converges uniformly on [a,b] to a function g, then the function f is differentiable and f'=g
- let ($f_n$) be a sequence of differentiable functions defined on the closed interval [a,b], and assume $(f'_n)$ converges uniformly to a function g on [a,b].  If there exists a point $x_0 \in [a,b]$ for which $f_n(x_0)$ is convergent, then ($f_n$) converges uniformly.  Moreover, the limit function f = lim $f_n$ is differentiable and satisfies f' = g

## 6.4 series of functions
- *term-by-term continuity thm* - let $f_n$ be continuous functions defined on a set A $\subseteq$ R and assume $\sum f_n$ converges uniformly on A to a function f.  Then, f is continuous on A.
- *term-by-term differentiability thm* - let $f_n$ be differentiable functions defined on an interval A, and assume $\sum f'_n(x)$ converges uniformly to a limit g(x) on A.  If there exists a point $x_0 \in [a,b]$ where $\sum f_n(x_0)$ converges, then the series $\sum f_n(x)$ converges uniformly to a differentiable function f(x) satisfying f'(x) = g(x) on A.  In other words, $f(x) = \sum f_n(x)$ and $f'(x) = \sum f'_n(x)$
- *Cauchy Criterion for uniform convergence of series* - A series $\sum f_n$ converges uniformly on A iff $\forall \epsilon > 0 \exists N \in N$ s.t. whenever n>m≥N, x in A $\|f_{m+1}(x) + f_{m+2}(x) + f_{m+3}(x) + ...+f_n(x)\| < \epsilon$
	- *Wierstrass M-Test* - For each n in N, let $f_n$ be a function defined on a set A $\subseteq$ R, and let $M_n > 0$ be a real number satisfying $\|f_n(x)\| ≤ M_n$ for all x in A.  If $\sum M_n$ converges, then $\sum f_n$ converges uniformly on A

## 6.5 power series
- power series f(x) = $\sum_{n=0}^\infty a_n x^n = a_0 + a_1 x_1 + a_2 x^2 + a_3 x^3 + ...$
- if a power series converges at some point $x_0 \in \mathbb{R}$, then it converges absolutely for any x satisfying \|x\|<\|$x_0$\|
- if a power series converges pointwise on the set A, then it converges uniformly on any compact set K $\subseteq$ A
	- if a power series converges absolutely at a point $x_0$, then it converges uniformly on the closed interval [-c,c], where c = \|$x_0$\|
	- *Abel's thm* - if a power series converges at the point x = R > 0, the the series converges uniformly on the interval [0,R].  A similar result holds if the series converges at x = -R
- if $\sum_{n=0}^\infty a_n x^n$ converges for all x in (-R,R), then the differentiated series $\sum_{n=0}^\infty n a_n x^{n-1}$ converges at each x in (-R,R) as well.  Consequently the convergence is uniform on compact sets contained in (-R,R).
	- can take infinite derivatives

## 6.6 taylor series
- *Taylor's Formula* $\sum_{n=0}^\infty a_n x^n = a_0 + a_1 x_1 + a_2 x^2 + a_3 x^3 + ...$
	- centered at 0: $a_n = \frac{f^{(n)}(0)}{n!}$
- *Lagrange's Remainder thm* - Let f be differentiable N+1 times on (-R,R), define $a_n = \frac{f^{(n)}(0)}{n!}.....$
- not every infinitely differentiable function can be represented by its Taylor series (radius of convergence zero)

# ch 7 - the Riemann Integral

## 7.2 def of Riemann integral
- *partition* of [a,b] is a finite set of points from [a,b] that includes both a and b
- *lower sum* - sum all the possible smallest rectangles
- a partition Q is a *refinement* of a partition P if $P \subseteq Q$ 
- if $P \subseteq Q$, then L(f,P)≤L(f,Q) and U(f,P)≥U(f,Q)
- a bounded function f on the interval [a,b] is *Riemann-integrable* if U(f) = L(f) = $\int_a^b f$
	- iff $\forall \epsilon >0$, there exists a partition P of [a,b] such that $U(f,P)-L(f,P)<\epsilon$
	- U(f) = inf{U(f,P)} for all possible partitions P
- if f is continuous on [a,b] then it is integrable

## 7.3 integrating functions with discontinuities
- if f:[a,b]->R is bounded and f is integrable on [c,b] for all c in (a,b), then f is integrable on [a,b]

## 7.4 properties of Integral
- assume f: [a,b]->R is bounded and let c in (a,b).  Then, f is integrable on [a,b] iff f is integrable on [a,c] and [c,b].  In this case we have $\int_a^b f = \int_a^c f + \int_c^b f.$F
- *integrable limit thm* - Assume that $f_n \to f$ uniformly on [a,b] and that each $f_n$ is integarble.  Then, f is integrable and $lim_{n \to \infty} \int_a^b f_n = \int_a^b f$.

## 7.5 fundamental theorem of calculus
1. If f:[a,b] -> R is integrable, and F:[a,b]->R satisfies F'(x) = f(x) for all x $\in$ [a,b], then $\int_a^b f = F(b) - F(a)$
2. Let f: [a,b]-> R be integrable and for x $\in$ [a,b] define G(x) = $\int_a^x g$.  Then G is continuous on [a,b]. If g is continuous at some point $c \in [a,b]$ then G is differentiable at c and G'(c) = g(c).

# overview
- convergence
	1. sequences
	2. series
	3. functional limits
		- normal, uniform
	4. sequence of funcs
		- pointwise, uniform
	5. series of funcs
		- pointwise, uniform
	6. integrability
- sequential criterion - usually good for proving discontinuous
	1. limit points
	2. functional limits
	3. continuity
	4. absence of uniform continuity
- algebraic limit theorem ~ scalar multiplication, addition, multiplication, division
	1. limit thm
	1. sequences
	2. series - can't multiply / divide these
	3. functional limits
	4. continuity
	6. differentiability
	7. ~integrability~
- limit thms
	- *continuous limit thm* - Let ($f_n$) be a sequence of functions defined on A that converges uniformly on A to a function f.  If each $f_n$ is continuous at c in A, then f is continuous at c
	- *differentiable limit theorem* - let $f_n \to f$ pointwise on the closed interval [a,b], and assume that each $f_n$ is differentiable.  If $(f'_n)$ converges uniformly on [a,b] to a function g, then the function f is differentiable and f'=g
		- convergent derivatives almost proves that $f_n \to f$
		- let ($f_n$) be a sequence of differentiable functions defined on the closed interval [a,b], and assume $(f'_n)$ converges uniformly to a function g on [a,b].  If there exists a point $x_0 \in [a,b]$ for which $f_n(x_0) \to f(x_0)$ is convergent, then ($f_n$) converges uniformly
	- *integrable limit thm* - Assume that $f_n \to f$ uniformly on [a,b] and that each $f_n$ is integarble.  Then, f is integrable and $lim_{n \to \infty} \int_a^b f_n = \int_a^b f$.
- functions are continuous at isolated points, but limits don't exist there
- uniform continuity: minimize $\|f(x)-f(y)\|$
- derivative doesn't have to be continuous
- integrable if finite amount of discontinuities and bounded

<!--link rel="stylesheet" type="text/css" href="collapse_working.css"-->
<!--script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.0/jquery.min.js"></script-->
<!--script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"></script-->