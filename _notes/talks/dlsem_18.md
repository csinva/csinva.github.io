# emergence theory of deep learning - Stefano Soatto

- information bottleneck method
  - introduced by Tishby, Pereira, Bialek
  - tradeoff between accuracy and complexity (compression) when summarizing (e.g. clustering) a random variable **X**, given **p(X, Y)** where Y observed
- papers
  - Emergence of Invariance and Disentangling in Deep Representations - Alessandro Achille, Stefano Soatto <https://arxiv.org/abs/1706.01350>
  - "Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks" - Pratik Chaudhari, Stefano Soatto <https://arxiv.org/abs/1710.11029>
- *representation* - any function of data that is useful for a task
  - you cannot create info in data - transforming data at best preserves information
  - ml function - function of future data constructed from past data that is useful for task regardless of nuisance factors
  - without task, raw data is best representation
- desiderata for an "optimal" representation
  - data x, task y, repr. z ~ p(z|x) - this represents a stochastic function
  - *sufficient* $I(z; y) = I(x; y)$
  - *minimal* $I(x;z)$ is minimal among inefficient z
  - *invariant to nuisances*: if $n \perp y$ then $I(n, z) = 0$
    - *maximally insensitive*: if $n \perp y$ then $I(n, z)$ is minimized
  - *maximally disentangled*: minimizes total correlation $KL[ p(z) | \prod_i p(z_i)]$
- for only sufficiency and minimality - Information Bottleneck, Tishby et al. 1999
  - $$\text{minimize}_{p(z|x)} I(x;z) \\ \text{s.t.} H(y|z)=H(y|x)$$
  - intractable except for exponential family so Tishby solves Lagrangian $\mathcal{L} = \underbrace{H_{p, q}(y|z)}_{cross-entropy} + \beta I(z;x)$
  - in fact, gives us invariances since *invariant* $\iff$ minimal
- layers - if last layer is sufficient, it is more minimal (invariant) than previous layers
  - because going through a layer can only lose info
- deep learning dataset D, 
  - $\underbrace{H_{p, q}(D|w)}_{cross-entropy} = H(D|\theta) + I(\theta; D|w) + KL(q|p) - \underbrace{I(D; w, \theta)}_{overfitting}$
    - easy to overfit by storing w completely
  - minimizing $\underbrace{H_{p, q}(D|w)}_{cross-entropy} + \underbrace{I(D; w, \theta)}_{overfitting}$ would minimize all the terms that we want, but second term is incomputable
  - instead SGVB (kingman and welling, 2015), minimize $\underset{q(w|D)}{arg min} \: \underbrace{H_{p, q}(D|w)}_{cross-entropy} + \beta \underbrace{I(D; w)}_{overfitting}$ 
    - looks like informaiton bottleneck, but slightly different
    - describes future data given past data
    - also can do the smae things using PAC_Bayes
  - SGD approximates this $\underset{q}{argmin} \: H_{p,q} - \beta H(q)$
    - amazing that SGD does this
  - for deep networks this and the normal information bottleneck are duals of each other
- one layer $z = wx$
  - $\underbrace{g[I(w; D)]}_{weights} \leq \underbrace{I(z; x) + TC(z)}_{activations} \leq \underbrace{g[I(w; D)] +c}_{weights}$
  - assume you get sufficiency (so probably need this for more layers)
  - weight minimality on past data bounds representation **minimality** of future data
  - corollary - less info in weights increases invariance and disentaglement of learned representation
  - corrolary - information bottleneck lagrangian and sgd are biased toward invariant and disentangled repr.
- sgd
  - sgd -> stochastic differential eqn
  - sgd doesn't minimize actual loss function but rather regularized thing above
  - sgd doesn't get to absolute minimum or local minimum
    - travels around on limit cycles
- summary
  - we want desiderata
  - practice - sgd / layers yield 
  - duality of representation - train set in weights, test set in activations



# what 4 year olds can do and AI can’t (yet) - alison gopnik

- gopnik 12 - reconstructing constructivism
  - how does genetics affeect the way that structure is built into the learning process of humans?
  - how does the sheer number of experiments that children do compare to that of modern robotics / machine learning algorithms?
  - if the model can change over time, what allows it to remain stable as it simultaneously searches different models and their parameters as well?
  - theories allow for *counterfactual inferences* - inferences about what happens if you intervene and/or do something new

# Deal or No Deal? End-to-End Learning for Negotiation Dialogues - mike lewis

- lewis_17 - Deal or No Deal? End-to-End Learning for Negotiation Dialogues
  - trained agents on negotiation task
- yarats_17 - Hierarchical Text Generation and Planning for Strategic Dialogue
  - want to adhere to human language - freeze these weights
- questions
  - what is the utility of having agents that can negotiate?
  - is there any way to build in a prior that incorporates a hierarchical linguistic structure (like syntax trees) into the speech model here?
  - is there any way to build in principles from fairness into these bots to avoid them learning biases?

# kevin murphy - graphical models

- liu_17_image_captioning
  - faster optimization for certain metrics
  - train on (image, caption) pairs - maximize ll
  - need better metrics (ex. SPICE), although hard to optimize
- vedantam_17_generative_visual_imagination
  - *visually grounded imagination* - create images of novel semantic concepts, even if you've never seen them before
  - use variational auto-encoders
  - new training objective
  - product-of-experts inference network
    - can handle partially specified concepts
- questions
  - is there a way to build one metric that combines SPICE and CIDEr (perhaps comparing some sort of hierarchical graph representation of a sentence)?
  - how do variational autoencoders compare to GANs for this task?
  - is there a way to learn concept hierarchies without predefining features?

# thomas funkhouser - "Data-Driven Methods for Matching, Labeling, and Synthesizing 3D Shapes"

- liu_18_gan_3d_modeling
  - using a gan to assist users in designing real-world shapes with a simple interface
  - SNAP - projects input into latent vector which feeds into gan
  - allows for interactive editing
- zeng_17_3dmatch
	- goal: establish correspondences between partial 3d data
	- extract local patches and learn correspondences w/ siamese network
- questions
  - Is there any way to cluster the kinds of things the GAN makes to make multiple different SNAP suggestions?
  - Are these correspondences precise enough to use for registration of medical images?
  - Are there different preferences a user can indicate to hep SNAP their input in different ways?

# dileep george - generative captchas and capsule net

- george_17_captcha
- sabour_17_capsules
  - want to make things invariant to pose
  - currently, training is much slower
  - what does a capsule do
    - outputs a vector for an object
    - magnitude of vector - probability of object being present
    - direction of vector - pose of object
    - learn weights with **dynamic routing** not backprop
      - sum of all weights to each capsule is 1
      - lower-level capusels sends input to higher-level capsule that "agrees" with its input
    - nonlinearity is vector $\to$ vector
- questions
  - How overfit is this captcha model to the task of digit recognition?
  - How much data was used to design the architecture of the network before training?
  - How can we speed up the training of capsule networks?

# bruno olshausen - perception in brains and machines

- questions
  - to what extent do modern cnns parallel the human visual system
  - how much of the human visual system is supervised vs unsupervised?
  - are there computational advantages of having a foveal image?

# alex kurakin - adversarial stuff

- elsayed_18_adversarial_human
- kannan_18_adversarial_logit
  - adversarial logit planning - try to get adversarial and clean image to have similar embeddings
- questions
  - if adversarial perturbation works on humans, does it really count as an adversarial example?
  - what is the intuition for how adversarial logit planning works?
  - how does adversarial logit planning affect the speed of training?

# abhinav gupta - unsupervised learning + video

- "Unsupervised Learning of Visual Representations using Videos" - unsupervised method: visual tracking provides supervision - 2 patches connected by track should have similar deep visual repr.


- learning by asking questions - try to be more sample efficient by asking for new data 
- transitive invariance for selv-supervised visual repr. learning
  - learn graph changing intra-instance, inter-instance, and transitive invariance
- questions
  - how effective is unsupervised learning in dealing with video data?
  - what are the major obstacles to bridge between video data and image data?
  - what do you think are the most promising methods of self-supervision in fields outside of computer vision?

# ryan adams

- "Mapping Sub-Second Structure in Mouse Behavior"
- "Composing graphical models with neural networks for structured representations and fast inference"
- questions
  - in the mouse paper, was there any way to support the movement modules with neural recording data?
  - how can we design better dimensionality techniques for temporal data using graphical models?
  - what tasks do you think are most promising for merging graphical models with neural networks?