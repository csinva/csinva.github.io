---
layout: notes
section-type: notes
title: Information Theory
category: stat
---

* TOC
{:toc}

# 1 - a measure of information
- *fundamental theorem of information theory* - it is possible to transmit information through a noisy channel at any rate less than channel capacity with an arbitrarily small probability of error
	- to achieve arbitrarily high reliability, it is necessary to reduce the transmission rate to the *channel capacity*
- uncertainty measure axioms
	1. H(1/M,...,1/M)=f(M) is a montonically increasing function of M
	2. f(ML) = f(M)+f(L) where M,L $$\in \mathbb{Z}^+$$
	3. *grouping axiom*
	4. H(p,1-p) is continuous function of p
- \$$H(p_1,...,p_M) = - \sum p_i log p_i = E[h(p_i)]$$
	- \$$h(p_i)= - log(p_i)$$
	- only solution satisfying above axioms
	- H(p,1-p) has max at 1/2
- three interpretations
	1. expectation of variable $$W=W(X)$$, which assumes the value $$-log(p_i)$$ with probability $$p_i$$
	2. minimum average number of "yes or no" questions required to determine result on one observation of X
	3. related to asymptotic behavior of sequence of i.i.d. random variables
- *lemma* - Let $$p_1,...,p_M$$ and $$q_1,...,q_M$$ be arbitrary positive numbers with $$\sum p_i = \sum q_i = 1$$. Then $$-\sum p_i log p_i \leq - \sum p_i log q_i$$. Only equal if $$p_i = q_i \: \forall i$$
	- intuitively, $$\sum p_i log q_i$$ is maximized when $$p_i=q_i$$, like a dot product
- $$H(p_1,...,p_M) \leq log M$$ with equality iff  all $$p_i = 1/M$$
- $H(X,Y) \leq H(X) + H(Y)$ with equality iff X and Y are independent
- $H(X,Y)=H(X)+H(Y|X) =H(Y)+H(X|Y)$
- $I(X,Y)=H(Y)-H(Y|X)$
- sometimes allow p=0 by saying 0log0 = 0
- information $I(x)=log_2 \frac{1}{p(x)}=-log_2p(x)$
- reduction in uncertainty (amount of surprise in the outcome)
- if the probability of event happening is small and it happens the info is large
    - entropy $H(X)=E[I(X)]=\sum_i p(x_i)I(x_i)=-\sum_i p(x_i)log_2 p(x_i)$
- higher entropy $\implies$ more uniform
- lower entropy $\implies$ more pure
- information gain $IG(X,Y)=H(Y)-H(Y|X)$
    - $H(Y|X)=\sum_j p(x_j) H(Y|X=x_j)$
    - $=-\sum_j p(x_j) \sum_i p(y_i|x_j) log_2 p(y_i|x_j)$

# 2 - noiseless coding
- parts
	1. random variable X taking on $$x_1,...,x_M$$ with probabilities $$p_1,...,p_M$$
	2. code alphabet = set $$a_1,...,a_D$$ . Each symbol $$x_i$$ is assigned to finite sequence of code characters called *code word* associated with $$x_i$$
	3. *objective* - minimize the average word length $$\sum p_i n_i$$ where $$n_i$$ is average word length of $$x_i$$
- code is *uniquely decipherable* if every finite sequence of code characters corresponds to at most one message
	- *instantaneous code* - no code word is a prefix of another code word