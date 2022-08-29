---
layout: notes
title: transformers
category: research
---


{:toc}

# papers

## high-performing

**nlp**

- attention is all you need ([vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) - initial transformer
  - encoder-decoder transformer for seq-to-seq
  - this paper has special encoder-decoder structure for translation (most new models don't)
  - [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) (by [Andrew Dai](https://twitter.com/iamandrewdai) and [Quoc Le](https://twitter.com/quocleix))
    - context vector is weighted sum of context vector at each word
- [ULMFiT](https://arxiv.org/abs/1801.06146) ([Jeremy Howard](https://twitter.com/jeremyphoward) and [Sebastian Ruder](https://twitter.com/seb_ruder))
- BERT ([devlin et al. 2018](https://arxiv.org/abs/1810.04805)) - semi-supervised learning (predict masked word - this is bidirectional) + supervised finetuning
  - [roberta](https://arxiv.org/abs/1907.11692)
- [ELMo](https://arxiv.org/abs/1802.05365) (by [Matthew Peters](https://twitter.com/mattthemathman) and researchers from [AI2](https://allenai.org/) and [UW CSE](https://www.engr.washington.edu/about/bldgs/cse)) - no word embeddings - train embeddings w/ bidirectional lstm (on language modeling)
- [XLNet](https://arxiv.org/abs/1906.08237)
- GPT-3 ([brown et al. 2020](https://arxiv.org/abs/2005.14165?2)) - identitical to GPT-2 except larger and replaces dense attention with sparse attention
  - GPT-2 ([radford et al. 2018](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf))
  - GPT ([radford et al. 2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf))
- gopher - basically gpt-3 with slight mods (replace layernorm by RMSnorm, different positional embeddings)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) (2022) - 540 Billion params
  - pathways hardware center allows for fast/efficient training
  - discontinuous improvements - at some point large model improves
  - prompt engineering: "Explain yourself" - lets it explain jokes
- [Chinichilla: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
  - for compute-optimal training, the model size and the number of training tokens should be scaled equally
- T0 ([sanh...rush, 2022](https://arxiv.org/pdf/2110.08207.pdf)) - multitask training enables better zero-shot generalization
  - [T5](https://jmlr.org/papers/volume21/20-074/20-074.pdf) (raffel...liu, 2020) -- text-to-text transfer transformer

**other**

- text-vision models
  - CLIP ([radford et al. 2021](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf)) - jointly train text/images
    - batch-based loss: encodings from same image/text pair should be close while encodings across different examples in the batch should be different
    - note: empirically works better with very large batch size
  - [dall-e 2](https://openai.com/dall-e-2/) (2022)
    - clip is foundation as generative model
    - generates text + image embeddings
    - "prior network" maps text embedding to image embedding
    - adds diffusion model
- vision
  -  [attention augmentation to resnet](https://arxiv.org/abs/1904.09925) for vision  (2020)
- multimodal
  - BEiT-3 ([2022](https://arxiv.org/abs/2208.10442)) - treat vision as language and large-scale multimodal training
- GATO: [A Generalist Agent](https://arxiv.org/abs/2205.06175) (2022) - single agent plays many different video games
  - different modalities are converted to tokens differently (e.g. image patches are fed through resnet)
- [spatial transformers](https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf )

##  adaptation / transfer

- basic approaches
  - **finetuning** ([peters et al. 2018](https://aclanthology.org/N18-1202/)) - train linear model on the embedding of the first token (usually an added `[CLS]` token)
  - finetune all parameters
  - adapter - between finetuning all layers, and just finetuning a new layer
    - add some new layers and retrain some specific things (all human choices)
  - prompting = few-shot learning = priming = in-context learning (starts with GPT)
    - limitation: can't exploit sets longer than the training window
  - misc
    - ablate some model weights by training a binary mask over model parameters (Zhao et al., 2020; Radiya-Dixit and Wang, 2020)
    - Zhang et al. (2020a) trains a “side” network that is fused with the pretrained model via summation
- few-shot papers
  - PatternExploiting Training (PET) -- Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference ([schick & schutze, 2021](https://aclanthology.org/2021.eacl-main.20.pdf))
    - **cloze questions** - same as masked language modeling: task is to replace some missing words
    - use cloze-question templates (e.g. it was "good" or "bad") to get soft labels for unlabeled data and then finetune on theses
  - LM-BFF [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/abs/2012.15723) (gao et al. 2020)
    - uses T5 to generate (i) template for the task (which might include a whole example or two) + (i) appropropriate label tokens in the vocabulary for the task (suffers from computationally intensive search + sub-optimal discrete space search)
  - [Cutting Down on Prompts and Parameters: Simple Few-Shot Learning with Language Models](https://arxiv.org/abs/2106.13353) (logan...sameer singh, eidel, 2021) -- finetuning in the few-shot setting can allow for much simpler prompts later (e.g. even null prompts)
  - [Adapting Language Models for Zero-shot Learning by Meta-tuning on Dataset and Prompt Collections](https://arxiv.org/abs/2104.04670) (zhong...dan klein, 2021)

## autoprompting

- [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://aclanthology.org/2020.emnlp-main.346/) (shin...sameer singh, 2020)
  - select prompts from a fixed set of tokens (resulting prompts are not coherent)
  - only work on MLM
  - elicit sentiment / factual knowledge
  - [Universal Adversarial Triggers for Attacking and Analyzing NLP](https://arxiv.org/abs/1908.07125) (wallace...sameer singh, 2022) - find input-agnostic sequences of tokens that trigger a model to produce a specific prediction when concatenated to any input from a dataset
  
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) (li & percy liang, 2021) -- optimizes in continuous space for language generation tasks
  - learn to map some parameters $\theta$ through and MLP to generate a starting hidden state $h_i$ -- never actually sends the prefix through the network 
  - [Control Prefixes for Parameter-Efficient Text Generation](https://arxiv.org/abs/2110.08329) (clive, cao, & rei, 2022) - allow for adapting the prefix to each input example
  
- DART [Differentiable Prompt Makes Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/abs/2108.13161) (zhang...chen, 2022)
  - reformulating NLP task into differentially optimizing the prompt template + target label (given a pre-trained model)
  - focus on smaller models (Roberta-large + GPT-2) + few training shots
  - fluency constraint to ensure association among prompt embeddings
  - P-Tuning -- [GPT Understands, Too](https://arxiv.org/abs/2103.10385) (liu et al. 2021) -- use LSTM to generate prompt embeddings (don't map to tokens)
- [Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification](https://arxiv.org/abs/2108.02035) (hu et al. 2021) -- add knowledge-base info into the prompt search
- [PTR: Prompt Tuning with Rules for Text Classification](https://arxiv.org/abs/2105.11259) (han et al. 2021) -- use logic rules to construct prompts with sub-prompts for many-class text classification
- [Learning How to Ask: Querying LMs with Mixtures of Soft Prompts](https://arxiv.org/abs/2104.06599) (qin & eisner, 2021); [github](https://github.com/hiaoxui/soft-prompts)
  - use continuous tokens and ensemble (don't map back to words)
- [WARP: Word-level Adversarial ReProgramming](https://arxiv.org/abs/2101.00121) (Hambardzumyan et al. 2021) - add continous tokens (don't map back to words) + some task-specific parameters for better generalization
- [KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction](https://arxiv.org/abs/2104.07650) (chen et al. 2021) -- incorporate relations, visualize learned prompt vectors with t-SNE
- misc
  - [SentiPrompt: Sentiment Knowledge Enhanced Prompt-Tuning for Aspect-Based Sentiment Analysis](https://arxiv.org/abs/2109.08306) -- use sentiment knowledge penalties in the prompt
  - [Meta-learning via Language Model In-context Tuning](https://arxiv.org/abs/2110.07814) (Chen et al. 2022) -- Given new task with new instruction
  - [Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm](https://arxiv.org/abs/2102.07350) (Reynolds & McDonell, 2021) -- define metaprompts as general wrappers around tasks e.g. “This problem asks us to”
- critiques of prompting
  - [Do Prompt-Based Models Really Understand the Meaning of their Prompts?](https://arxiv.org/abs/2109.01247) (webson & pavlick, 2022) -- - models can learn fine with prompts that are intentionally irrelevant

## model chaining

**notes from this [thread](https://twitter.com/iraphas13/status/1551959289023016967) on chaining models together**:

- steering
  - overviews
    - [AI Chains: Transparent and Controllable Human-AI Interaction by Chaining Large Language Model Prompts](https://arxiv.org/abs/2110.01691) (wu et al. 2022) - chaining LLM steps together: output of one step becomes the input for the next
      - interactive system where users can modify chains + their intermediate results
    - [Language Model Cascades](https://arxiv.org/abs/2207.10342) - treat chaining models as probabilistic programs
      - use a probabilistic-programming language (PPL) to define a joint probability model on string-valued random variables, parameterized using LMs, and then condition this model on string-valued observations in order to compute a posterior over string-valued unknowns
      - PPLs extend probabilistic graphical models to support more complex joint distributions whose size and “shape” can itself be stochastic
        - e.g., a graph unrolled for a random number of iterations, until a data-dependent stopping criterion is met
        - variables are all text: questions $Q$, answers $A$, and intermediate thoughts $T$
  - posthoc
    - [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) (wei et al. 2022)
      - in few-shot prompts, don't just provide answer but also reasoning
      - model output then provides reasoning + answer
    - Scratchpads [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114) (nye et al. 2021)
    - selection inference ([creswell et al. 2022](https://arxiv.org/abs/2205.09712)) - generate set of facts, then iteratively generate inferences from the facts to yield the final answer
    - least-to-most prompting ([zhou...quoc le et al. 2022](https://arxiv.org/abs/2205.10625)) - prompt LLM with context showing how to reduce into subproblems; then LLM sequentially solves the subproblems, using the previous answers
  - training
    - verifiers ([cobbe et al. 2021](https://arxiv.org/abs/2110.14168)) - train model to judge whether an answer and thought are likely to be “valid”
    - maieutic prompting ([jung et al. 2022](https://arxiv.org/abs/2205.11822)) - generate a tree of all explanation of the form "True, because...", "False, because..." then query LLM with these as prompts
      - then use Max-SAT to try to satisfy as many relations between the model explanations as possible to come up with the true answer
    - subgoal search ([czechowski et al. 2021](https://t.co/PCR4yexHti)) - train model to generate subgoals then solve them in a graph
    - STaR “Self-taught reasoner” ([zelikman...goodman, 2022](https://arxiv.org/abs/2203.14465))
      - first, finetune on observed $(Q, T, A)$ triplets
      - then, impute unknown $T_i$ given dataset of pairs $(Q_i, A_i)$ by sampling until finding a $T_i$ which leads to the correct answer
  - robotics-specific
    - zero-shot planning [arxiv.org/abs/2201.07207](https://arxiv.org/abs/2201.07207)
    - socratic models [arxiv.org/abs/2204.00598](https://arxiv.org/abs/2204.00598)
    - Inner Monologue [arxiv.org/abs/2207.05608](https://arxiv.org/abs/2207.05608)
    - [global workspace](https://arxiv.org/abs/2103.01197)
- more efficient training
  - natural language feedback ([scheurer et al. 2022]())
    - human feedback for learning makes it much more efficient
- augmenting
  - add retrieved data to context
    - [A Neural Corpus Indexer for Document Retrieval](https://arxiv.org/abs/2206.02743) - train model to directly spit out document IDs given queries
    - [lamda](https://arxiv.org/abs/2201.08239) - allows google search to add world info (in a dialog model)
      - this was the model that sparked the controversy about consciousness 🤔
    - [webgpt](https://arxiv.org/abs/2112.09332) - allows google search to add world info
    - [RLPG](https://arxiv.org/abs/2206.12839) - retrieves functions from the repo, for code-completion
    - [REALM](arxiv.org/abs/2002.08909) - retrieves document chunks from corpus and adds them to context, for open-domain QA
    - [memory-assisted prompt-editing](https://arxiv.org/abs/2201.06009) - allows model to "save things to memory" that get added to prompt when needed
  - increasing attendable context size with augmented models
    - RETRO [arxiv.org/abs/2112.04426](https://arxiv.org/abs/2112.04426) - nearest neighbors to model's input are retrieved, encoded, and conditioned on with chunked cross-attention 
    - memorizing transformers [arxiv.org/abs/2203.08913](https://arxiv.org/abs/2203.08913) - knn-based learned indexing + retrieval at training time. at input time, you just need to index the entire context and the model will be able to use it
- task-specific
  - [instructGPT](https://arxiv.org/abs/2203.02155) / [FLAN](https://arxiv.org/abs/2109.01652) - finetune on instructions to follows instructions
  - [MINERVA: Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858) - train on well-parsed, domain-specific data (math arxiv) to solve math-reasoning problems
  - autoformalization [arxiv.org/abs/2205.12615](https://arxiv.org/abs/2205.12615) - translating from natural language math to formal language
  - program synthesis [arxiv.org/abs/2108.07732](https://arxiv.org/abs/2108.07732) - formalize natural language into runnable code

## model editing

- [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (meng et al. 2022)
  - causal intervention for identifying neuron activations that are decisive in a model’s factual predictions
  - modify feedforward weights to update specific factual associations using Rank-One Model Editing (ROME)

## transformer circuits

**[thread](https://transformer-circuits.pub/2021/framework/index.html) (elhage...olah, 2021)**

- all layers are same dimension and each attention block **adds** a vector to it
- Although they’re parameterized as separate matrices, $W_O W_V$ and $W_Q^T W_K$ can always be thought of as individual, low-rank matrices
  - $x \in \mathbb R^{d_{embed} \times d_{sequence}}$: $d_{embed}$ can be hundreds - tens of thousands 
  - $W_Q, W_K, W_V \in \mathbb R^{d_{attn} \times d_{embed}}$
    - $W_Q^TW_k \in \mathbb R ^{d_{embed} \times d_{embed}}$
  - $W_O \in \mathbb R^{d_{embed} \times d_{attn}}$: projects attention values back to embedding dimention
    - $W_O W_V \in \mathbb R ^{d_{embed} \times d_{embed}}$
  - $W_E \in \mathbb R^{d_{embed} \times d_{vocab}}$ embeds initial tokens and $W_U \in \mathbb R^{d_{vocab} \times d_{embed}}$ undoes the embedding
    - $d_{vocab}$ can be very large, e.g. 50k
  - $A = \text{softmax}(x^TW_Q^TW_kx) \in \mathbb R^{d_{sequence} \times d_{sequence}}$
- if we have a 0-layer net (e.g. predict next token with linear layer given current token), we just learn bigram log-likelihood
- 2 circuits
  - QK circuit determines which "source" token the present "destination" token attends back to and copies information from
    - $W_{E}^{T} W_{Q}^{T} W_{K} W_{E} \in \mathbb R ^{d_{vocab} \times d_{vocab}}$
  - OV circuit describes what the resulting effect on the "out" predictions for the next token is
    - $W_{U} W_{O} W_{V} W_{E} \in \mathbb R ^{d_{vocab} \times d_{vocab}}$
- if a single head increases the probability of both `keep… in mind` and `keep… at bay`, it *must* also increase the probability of `keep… in bay` and `keep… at mind`
- **induction heads** search previous examples of present token
  - If they don't find it, they attend to the first token and do nothing
  - if they do find it, they then look at the *next* token and copy it. This allows them to repeat previous sequences of tokens, both exactly and approximately
  - sometimes can do some kind of "fuzzy" matching
- tensor/kronecker product $\bigotimes$:
  - Left-right multiplying: Multiplying $x$ by a tensor product $A \otimes W$ is equivalent to simultaneously left and right multiplying: $(A \otimes W) x=A x W^{T}$
  - When we add them, it is equivalent to adding the results of this multiplication: $\left(A_{1} \otimes W_{1}+A_{2} \otimes W_{2}\right) x=A_{1} x W_{1}^{T}+A_{2} x W_{2}^{T}$ 

**[Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html)**

- replacing activation function with softmax linear unit increases fraction of MLP neurons which are "interpretable", i.e. correspond to meaningful features
  - however, may “hide” some non-neuron-aligned features by decreasing their magnitude and then later recovering it with LayerNorm
- the presence of nonlinear activation functions createse an incentive for features to align with this basis and not get superposed
  - if the gains to sparse coding are large enough, this incentive will get overwhelmed
- ways to combat polysemanticity
  - activation sparsity
  - lateral inhibition / co-occurrence sparsity
  - weight sparsity
  - superlinear activation functions
  - increase neurons per param
- $\text{SoLU}(x) = x \cdot \text{softmax}(x)$
  - adds lateral inhibition, superlinearity, approximate sparsity
  - changes GeLU, which is approximately $\text{sigmoid}(1.7x) \cdot x$
  - just changing to SoLU decrease performance, had to add LayerNorm afterwards

## mixture of experts (MoE)

- note: nowadays often the "experts" are different MLPs following the self-attention layers
- non-specialized experts
  - Early versions ([Jacobs et al., 1991](https://ieeexplore.ieee.org/abstract/document/6797059)) had independent feed-forward networks serving as experts
  - Recent MoE models ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)) have been studied with token-based routing with backprop
  - GShard [Lepikhin et al. (2021)](https://arxiv.org/abs/2006.16668), which appplies this concept to machine translation
  - Switch transformers [Fedus et al. (2022)](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf) simplifies the architecture to activation of only one expert per layer
  - Base Layers [Lewis et al. (2021)](https://proceedings.mlr.press/v139/lewis21a.html) - find an alternative approach to routing by formulating it as a linear assignment problem
  - Hash layers [Roller et al. (2021)](https://arxiv.org/abs/2106.04426) use a fixed hash as the gating function
- specialized experts
  - DEmix Layers [Gururangan et al.](https://arxiv.org/abs/2108.05036) (2022) --  DEMix layers – placed in the feedforward layers of the Transformer – contain experts which specialize on specific domains. Routing at train time is determined only by the domain label, but all experts are activated at inference time and mixed according to weights estimated from a validation set
  - [Pfeiffer et al. (2022)](https://arxiv.org/abs/2205.06266) - multilingual expert model with language-specific routing
  - task-level MoE [Kudugunta et al. (2021](https://arxiv.org/abs/2110.03742)) -- multi-task expert model with task-specific routing
  - ELMS -- Branch-Train-Merge ([li et al. 2022](https://arxiv.org/abs/2208.03306))
    - parallel language model of smaller expert LMs
    - each can be added/removed, ensembled, or parameter-averaged at any time for efficient scaling and rapid customization
    - improves perplexities, when controlling for training cost
      - require expert domain specialization

## causal inference

- [InferBERT: A Transformer-Based Causal Inference Framework for Enhancing Pharmacovigilance](https://www.frontiersin.org/articles/10.3389/frai.2021.659622/full) (2021) - learn + test feature relationships from attention weights
- [CausaLM: Causal Model Explanation Through Counterfactual Language Models | Computational Linguistics](https://direct.mit.edu/coli/article/47/2/333/98518/CausaLM-Causal-Model-Explanation-Through) (2021) - produce example-level causal model explanations using models finetuned on auxiliary adversarial tasks derived from the causal graph of the problem

# basics

- **attention** = vector of importance weights
  - to predict or infer one element, such as a pixel in an image or a word in a sentence, we estimate using the attention vector how strongly it is correlated with (or “*attends to*” other elements and take the sum of their values weighted by the attention vector as the approximation of the target
- vanilla transformer: multihead attention, add + norm, position-wise ffn, add + norm
- self-attention layer [implementation](https://github.com/mertensu/transformer-tutorial) and [mathematics](https://homes.cs.washington.edu/~thickstn/docs/transformers.pdf)

## mathematical overview of transformers ([Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238?utm_source=substack&utm_medium=email))

- tasks
  - *sequence modeling*: learn $p(x)$, usually factorized as $p(x_i|x_1,...,x_{i-1})$
  - *sequence-to-sequence*: learn $p(z|x)$, e.g. transalation, speech-to-text, question answering
- preprocessing
  - embedding matrix takes in one-hot tokens and linearly maps them to a vector
  - positional embedding of a token is usually added to the token embedding to form a token’s initial embedding
- attention types
  - *Bidirectional / unmasked self-attention* - primary/context vectors are the same
  - *Unidirectional / masked self-attention* - mask scores from before a given word
  - *Cross-attention* - primary/context vectors can come from different places
- non-attention
  - layernorm: controls mean/variance of activations
    - RMSnorm: simpler version, sets mean/offset to zero
- unembedding
  - linear layer (with softmax) that outputs size of original vocab
    - sometimes fixed to be transpose of the embedding matrix
- predictions
  - predict next word using single linear layer on hidden state from previous word
  - finetune classification head often only using linear layer on first token from sequence

- architectures
  - initially, encoder-decoder was common, but now often no decoder

## visual explanation (notes on article by jay allamar)

- **self-attention ** - layer that lets word learn its relation to other layers
  - for each word, want score telling how much importance to place on each other word (queries $\cdot$ keys)
  - we get an encoding for each word
    - the encoding of each word returns a weighted sum of the values of the words (the current word gets the highest weight)
    - softmax this and use it to do weighted sum of values![Screen Shot 2019-08-17 at 2.51.53 PM](../assets/attention.png)
  - (optional) implementation details
    - **multi-headed attention** - just like having many filters, get many encodings for each word
      - each one can take input as the embedding from the previous attention layer
    - **position vector** - add this into the embedding of each word (so words know how far apart they are) - usually use sin/cos rather than actual position number
    - **padding mask** - add zeros to the end of the sequence
    - **look-ahead mask** - might want to mask to only use previous words (e.g. if our final task is decoding)
    - **residual + normalize** - after self-attention layer, often have residual connection to previous input, which gets added then normalized
  - decoder - each word only allowed to attend to previous positions
  - 3 components
    - queries
    - keys
    - values
- **attention**
  - encoder reads input and ouputs context vector after each word
  - decoder at each step uses a different weighted combination of these context vectors
    - specifically, at each step, decoder concatenates its hidden state w/ the attention vector (the weighted combination of the context vectors)
    - this is fed to a feedforward net to output a word
    - ![Screen Shot 2019-04-11 at 7.57.14 PM](../assets/nmt.png)
  - at a high level we have $Q, K, V$ and compute $softmax(QK^T)V$
    - instead could simplify it and do $softmax(XX^T)V$ - this would then be based on kernel
- **transformer**
  - uses many self-attention layers
  - many stacked layers in encoder + decoder (not rnn: self-attention + feed forward)
  - details
    - initial encoding: each word -> vector
    - each layer takes a list of fixed size (hyperparameter e.g. length of longest sentence) and outputs a list of that same fixed size (so one output for each word)
      - can easily train with a masked word to predict the word at the predicted position in the encoding
  - multi-headed attention has several of each of these (then just concat them)