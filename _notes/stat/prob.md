---
layout: notes
section-type: notes
title: Probability
category: stat
---
* TOC
{:toc}

# Properties
- Mutually Exclusive: P(AB)=0
- Independent: P(AB) = P(A)P(B)
- Conditional: P(A | B) = $\frac{P(AB)}{P(B)}$

# Measures
- $E[X] = \int P(x)x dx$
- $V[X] = E[(x-\mu)^2] = E[x^2]-E[x]^2$
	- for unbiased estimate, divide by n-1
- $Cov[X,Y] = E[(X-\mu_X)(Y-\mu_Y)] = E[XY]-E[X]E[Y]$
	- $Cor(Y,X) = \rho = \frac{Cov(Y,X)}{s_xs_y}$
	- $(Cor(Y,X))^2 = R^2$
	- Cov is a measure of how much 2 variables change together
- linearity
	- $Cov(aX+bY,Z) = aCov(X,Z)+bCov(Y,Z)$
	- $V(a_1X_...+a_nX_n) =  \sum_{i=1}^{n}\sum_{j=1}^{n}a_ia_jcov(X_i,X_j)$
	- if $X_1,X_2$ independent, $V(X_1-X_2) = V(X_1) + V(X_2)$
- $f(X), X=v(Y), g(Y) = f(v(Y))$ |$\frac{d}{dy}g^{-1}(y)$|
- $g(y_1,y_2) = f(v_1,v_2)|det(M)|$ where M in row-major is $\frac{\partial v1}{y1}, \frac{\partial v1}{y2} ...$
- $Corr(aX+b,cY+d) = Corr(X,Y)$ if a and c have same sign
- $E[h(X)] \approx h(E[X])$
- $V[h(X)] \approx h'(E[X])^2 V[X]$
- skewness = $E[(\frac{X-\mu}{\sigma})^3]$

# Moment-generating function
- $M_X(t) = E(e^{tX})$
- $E(X^r) = M_X ^ {(r )} (0)$
- sometimes you can use $ln(M_x(t))$ to find $\mu$ and $V(X)$
- Y = aX+b $-> M_y(t) = e^{bt}M_x(at)$
- Y = $a_1X_1+a_2X_2 \to M_Y(t) = M_{X_1}(a_1t)M_{X_2}(a_2t)$ if $X_i$ independent
- probability plot  - straight line is better - plot ([100(i-.5)/n]the percentile, ith ordered observation)
- ordered statistics - variables $Y_i$ such that $Y_i$ is the ith smallest
- If X has pdf f(x) and cdf F(x), $G_n(y) = (F(y))^n$, $g_n(y) = n(F(y))^{n-1}f(y)$
- If joint, $g(y_1,...y_n) = n!f(y_1)...f(y_n)$	
- $g(y_i) = \frac{n!}{(i-1)!(n-i)!}(F(y_i))^{i-1}(1-F(y_i))^{n-1}f(y_i)$

# Distributions
- Bernoulli: 
$f(x)= 
\begin{cases}
    1,& \text{if } 0\leq x\leq p\\
    0,              & \text{otherwise}
\end{cases}$

- Binomial: 
$f(n,p)= 
\begin{cases}
    {n \choose p} p^x (1-p)^{n-x},& \text{if } 0\leq x\leq p\\
    0,              & \text{otherwise}
\end{cases}$

- Gaussian: 
$f(\lambda)= 
\begin{cases}
    \frac{1}{\lambda}e^{\lambda t},& \text{if } 0\leq x\leq p\\
    0,              & \text{otherwise}
\end{cases}$