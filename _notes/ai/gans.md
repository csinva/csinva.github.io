---
layout: notes
section-type: notes
title: GANS
category: ai
---

* TOC
{:toc}

# GAN original
- might not converge
- *generative adversarial network*
- goal: want G to generate distribution that follows data
	- ex. generate good images
- two models
	- *G* - generative
	- *D* - discriminative
- G generates adversarial sample x for D
	- G has prior z
	- D gives probability p that x comes from data, not G
		- like a binary classifier: 1 if from data, 0 from G
	- *adversarial sample* - from G, but tricks D to predicting 1
- training goals
	- G wants D(G(z)) = 1
	- D wants D(G(z)) = 0
		- D(x) = 1
	- converge when D(G(z)) = 1/2
	- G loss function: $G = argmin_G log(1-D(G(Z))$
	- overall $min_g max_D$ log(1-D(G(Z))
- training algorithm
	- in the beginning, since G is bad, only train  my minimizing G loss function
	- later
		```
		for 
			for
				max D by SGD
			min G by SGD
		```
	