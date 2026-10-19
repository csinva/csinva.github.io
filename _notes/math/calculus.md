---
layout: notes
title: calculus
subtitle: A quick reference for single-variable and multivariable calculus.
category: math
---

{:toc}

# taylor expansion
- taylor expansion: $$f(x) \approx \cblue{f(x_0)} + \corange{\frac{f'(x_0)}{1!}(x-x_0)} + \cgreen{\frac{f''(x_0)}{2!}(x-x_0)^2} + ...$$
  - <span class="cblue">0th-order</span> + <span class="corange">1st-order</span> + <span class="cgreen">2nd-order</span> terms


# Single-variable calculus
Derivatives:

$\frac{d}{dx}x^n = nx^{n-1}$

$\frac{d}{dx}a^x = a^{x}\ln(a)$

$\frac{d}{dx}\ln(x) = 1/x$

$\frac{d}{dx}\tan(x)= sec^2(x)$

$\frac{d}{dx}cot(x)= -csc^2(x)$

$\frac{d}{dx}sec(x)= sec(x)\tan(x)$

$\frac{d}{dx}csc(x)= -csc(x)cot(x)$

$$\int \tan = \ln\vert sec\vert $$<!-- -->

$$\int cot = \ln\vert \sin\vert $$<!-- -->

$$\int sec = \ln\vert sec+\tan\vert $$<!-- -->

$$\int csc = \ln\vert csc-cot\vert $$<!-- -->

$\int \frac{du}{\sqrt{a^2-u^2}} = \sin^{-1}(\frac{u}{a})$

$\int \frac{du}{u\sqrt{u^2-a^2}} = \frac{1}{a}sec^{-1}(\frac{u}{a})$

$\int \frac{du}{a^2+u^2} = \frac{1}{a} \tan^{-1}(\frac{u}{a})$

Continuous: left limit = right limit = value

Differentiable: continuous and no sharp points / asymptotes

L'Hospital's - for indeterminate forms: $$(\frac{f(x)}{g(x)})' = \frac{f'(x)}{g'(x)}$$

Integration by parts: $\int{udv}=uv-\int{duv}$, LIATE

Expansions:

$e^x = \sum{\frac{x^n}{n!}}$

$\sin(x) = \sum_0^\infty{\frac{(-1)^n x^{2n+1}}{(2n+1)!}}$

$\cos(x) = \sum_0^\infty{\frac{(-1)^n x^{2n}}{(2n)!}}$

Geometric Sum: $a_{1st}\frac{1-r^{n+1}}{1-r}$

# Multivariable calculus
- Polar: r,$\theta$,z
- Spherical: $\rho,\theta,\phi$
- Clairut's Thm: Conservative function $f_{xy}=f_{yx}$
- *Lagrangian* - solves minimize <span class="cblue">f</span> subject to <span class="corange">g = c</span>
	- solution will always be *tangent* to f
	- $\cblue{\nabla f} = \corange{\lambda \nabla g}$ - gives us n constraints
	- remember g = c is a constraint too
	- to do this efficiently, define the *Lagrangian* $L(x, \lambda) = \cblue{f} - \corange{\lambda \cdot g}$
		- taking deriv wrt $\lambda$ and setting = 0 enforces g = c
		- taking deriv wrt other variables and setting = 0 enforces other conditions
		- therefore final eq just becomes $\nabla L = 0$