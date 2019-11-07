---
title: Chandan nn weights pres
separator: '----'
verticalSeparator: '---'
typora-copy-images-to: ./assets_files
revealOptions:
    transition: 'slide'
	transitionSpeed: 'fast'
---

<h1>  what do neural network weights reliably learn? </h1>
<h3> chandan singh </h3> 
## why analyze neural net weights?

- neural nets learn useful representations
- understanding weights might help with engineering
- might help us understand the brain
- approaches: neuroscience learning rules, unsupervised learning, NN analysis, feature engineering

## related work

- engineering strong feature representations (mallat 12, huang et al. 06)
- sparse coding (olshausen & field 96)
- analyzing weight matrices (denil et al. 13, martin et al. 18)
- analyzing neural activations (tishby et al. 15, 17)

## a layer can learn its input

$$W^{\text{final}} = W^{\text{init}} + \color{cyan}{W^{\text{proj}}}\\\ \color{cyan}{W^{\text{proj}}} \in col(X)$$

this is a consequence of the learning rule:

$$Y = g(W X)$$

$$\frac{\partial L} {\partial W} = \frac{\partial L} {\partial g} (W X) \cdot X$$

## weights sometimes grow (especially when using ADAM)

<img src="assets_files/frob_norm.png"  class="invert">

$\implies \color{cyan}{W^{\text{proj}}}$ dominates

# 1st layer weight viz (optimizer = adam)

<img src="assets_files/adam_lr.png"  class="invert">

## 1st layer weight viz (optimizer = sgd)

<img src="assets_files/sgd_lr.png"  class="invert">

## mnist sparse dictionary ($\lambda$ = 10)

<img src="assets_files/bases_iters=60000_alpha=10.png"  class="invert">

## mnist sparse dictionary ($\lambda$ = 100)

<img src="assets_files/bases_iters=60000_alpha=100.png"  class="invert">

# which layer?

## why the first layer?: training mlps

|    |  Linear Classifier    |  MLP-2 First-Layer    | MLP-2 Last-layer
| -- | -- | -- |
| MNIST |   0.92   |  **0.96**    | 0.90

## why the first layer?: training mlps

|    |  Linear Classifier    | Linear + Lenet First-layer
| -- | -- |
| MNIST |   0.92   | **0.98**

## 1st and last layer norms grow

<img src="assets_files/vgg11.png" class="invert">



# different nets learn the same thing

## adam weights can be pruned more

<img src="assets_files/mnist_pruning.png" class="invert">

## (val) preds correlate

<img src="assets_files/pred_corrs_val_full.png" class="invert">


## top k match

<img src="assets_files/topk.png" class="invert">

# memorization

## quantifying memorization

<img src="assets_files/coef.png" class="invert" height="10%">



## memorization is stable

<img src="assets_files/stability.png" class="invert">

# hyperparameters that increase memorization

- **ADAM over SGD**
- smaller batch size
- larger learning rate
- larger width
- etc.

## how does ADAM cause memorization?

- ADAM disproportionately increases first layer's learning rate
- increasing only first layer's learning rate qualitatively reproduces ADAM

## going deeper: memorization in deep cnns

<img src="assets_files/cnns.png" class="invert">

# linear experiments

## bias vs var

<img src="assets_files/Screen Shot 2019-07-29 at 11.51.53 AM.png" class="invert">



## mse changes based on distr

<img src="assets_files/Screen Shot 2019-07-29 at 11.53.33 AM.png" class="invert">



## mse with pcs

<img src="assets_files/Screen Shot 2019-07-29 at 11.53.41 AM.png" class="invert">



## logistic regression with cvs (score is mse)

<img src="assets_files/Screen Shot 2019-07-29 at 11.53.55 AM.png" class="invert">



## linear regression with cvs (score is mse)

<img src="assets_files/Screen Shot 2019-07-29 at 11.54.10 AM.png" class="invert">