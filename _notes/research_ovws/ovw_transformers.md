---
layout: notes
title: transformers
category: research
---


{:toc}

# papers

## high-performing

**nlp** (see also [this link](https://medium.com/nlplanet/a-brief-timeline-of-nlp-from-bag-of-words-to-the-transformer-family-7caad8bbba56))

- attention is all you need ([vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) - initial transformer
  - encoder-decoder transformer for seq-to-seq
  - this paper has special encoder-decoder structure for translation (most new models don't)
  - [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) (dai & quoc le, 2015)
    - context vector is weighted sum of context vector at each word
- [ULMFiT](https://arxiv.org/abs/1801.06146) ([Jeremy Howard](https://twitter.com/jeremyphoward) and [Sebastian Ruder](https://twitter.com/seb_ruder))
- BERT ([devlin et al. 2018](https://arxiv.org/abs/1810.04805)) - semi-supervised learning (predict masked word - this is bidirectional) + supervised finetuning
  - [roberta](https://arxiv.org/abs/1907.11692)
- [BART: Denoising sequence-to-sequence pre-training](https://arxiv.org/abs/1910.13461) (lewis et al. 2019) - generalizes BERT - train by (1) corrupting text then (2) reconstruct the original text
- [ELMo](https://arxiv.org/abs/1802.05365) (by [Matthew Peters](https://twitter.com/mattthemathman) and researchers from [AI2](https://allenai.org/) and [UW CSE](https://www.engr.washington.edu/about/bldgs/cse)) - no word embeddings - train embeddings w/ bidirectional lstm (on language modeling)
- XLNet ([yang...quoc le, 2020](https://arxiv.org/abs/1906.08237))
- GPT-3 ([brown et al. 2020](https://arxiv.org/abs/2005.14165?2)) - identitical to GPT-2 except larger and replaces dense attention with sparse attention
  - sizes: largest ha 175B params, 96 layers, 96 heads in each layer, head with dim 128, vocab size ~50k
  - GPT-2 ([radford et al. 2018](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf))
  - GPT ([radford et al. 2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf))

- [Gopher](https://arxiv.org/abs/2112.11446) - basically gpt-3 with slight mods (replace layernorm by RMSnorm, different positional embeddings)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) (beltagy, peters, & cohan, 2020) - processes very long contexts
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) (google, 2022) - 540 Billion params
  - pathways hardware center allows for fast/efficient training
  - discontinuous improvements - at some point large model improves
  - prompt engineering: "Explain yourself" - lets it explain jokes
- [Chinchilla: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
  - for compute-optimal training, the model size and the number of training tokens should be scaled equally
- T0 ([sanh...rush, 2022](https://arxiv.org/pdf/2110.08207.pdf)) - multitask training enables better zero-shot generalization
  - T5 ([raffel...liu, 2020](https://jmlr.org/papers/volume21/20-074/20-074.pdf)) -- text-to-text transfer transformer
- more efficient training
  - natural language feedback ([scheurer et al. 2022]())
    - human feedback for learning makes it much more efficient
    - [Can language models learn from explanations in context?](https://arxiv.org/abs/2204.02329) (lampinen et al. 2022)
  - [instructGPT](https://arxiv.org/abs/2203.02155) / [FLAN](https://arxiv.org/abs/2109.01652) - finetune on instructions to follows instructions
  - ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators ([clark...quoc le, chris manning, 2020](https://arxiv.org/abs/2003.10555))
    - more efficient: instead of standard masked training, have generator-discriminator setup for "token detection"
    - generator replaces many masked tokens with plausible samples (all in one forward pass) - train with MLM
    - discriminator tries to guess which tokens were the masked ones - this is the main model that gets used
- dialog
  - [GODEL: Large-Scale Pre-Training for Goal-Directed Dialog](https://arxiv.org/abs/2206.11309) (baolin peng, galley, ..., gao , 2022) - add grounded pre-training
  - [Deal or No Deal? End-to-End Learning for Negotiation Dialogues](https://arxiv.org/abs/1706.05125) (lewis...batra, 2017) - controversial FB paper where agents "make up their own language"
    - this is pre-transformers


**other**

- text-vision models
  - CLIP ([radford et al. 2021](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf)) - jointly train text/images
    - batch-based loss: encodings from same image/text pair should be close while encodings across different examples in the batch should be different
    - note: empirically works better with very large batch size
  - DALL-E 2 ([OpenAI, 2022](https://openai.com/dall-e-2/))
    - clip is foundation as generative model
    - generates text + image embeddings
    - "prior network" maps text embedding to image embedding
    - adds diffusion model
  - BEiT-3 ([2022](https://arxiv.org/abs/2208.10442)) - treat vision as language and large-scale multimodal training
    - outperforms [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) (2022), which uses more domain knowledge to connect vision & language

- interacting with computer

  - [ACT-1: Transformer for Actions](https://www.adept.ai/act)

- vision
  - here, people often call image patches "tokens"
  - [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (he...dollar, girshick, 2021) - BERT-style training
    -  speed up by not applying encoder to mask tokens + adding mask to a lot of the data (like 75%)
    -  really good results without much data
  - [attention augmentation to resnet](https://arxiv.org/abs/1904.09925) for vision (bello...quoc le, 2020)
- GATO: [A Generalist Agent](https://arxiv.org/abs/2205.06175) (2022) - single agent plays many different video games
  - different modalities are converted to tokens differently (e.g. image patches are fed through resnet)
- [MINERVA: Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858) - train on well-parsed, domain-specific data (math arxiv) to solve math-reasoning problems
- CODEX [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (2021)
  - [Repair Is Nearly Generation: Multilingual Program Repair with LLMs](https://arxiv.org/abs/2208.11640) (joshi et al. 2022) 
  - [Improving automatically generated code from Codex via Automated Program Repair](https://arxiv.org/abs/2205.10583) (fan et al. 2022) - use automated program repair to tweak codex outputs to make them better
  - [Automatic Program Repair with OpenAI's Codex: Evaluating QuixBugs](https://arxiv.org/abs/2111.03922) (prenner & robbes, 2021)
    - use prompt like:
      ```python
      ### fix the bug in the following function
      <buggy function and/or docstring here>
      ### fixed function
      ```
- autoformalization [arxiv.org/abs/2205.12615](https://arxiv.org/abs/2205.12615) - translating from natural language math to formal language
- program synthesis [arxiv.org/abs/2108.07732](https://arxiv.org/abs/2108.07732) - formalize natural language into runnable code
- [spatial transformers](https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf )
- speculative [foundation models paper](https://arxiv.org/abs/2108.07258) (stanford, 2022)

**pre-transformer nlp models**

- rnns
  - when training rnn, accumulate gradients over sequence and then update all at once
  - **stacked rnns** have outputs of rnns feed into another rnn
  - bidirectional rnn - one rnn left to right and another right to left (can concatenate, add, etc.)
- standard seq2seq
  - encoder reads input and outputs context vector (the hidden state)
  - decoder (rnn) takes this context vector and generates a sequence

##  adaptation / transfer

*These are transformer-specific. For more general notes, see [📌 transfer learning](https://csinva.io/notes/research_ovws/ovw_transfer_learning.html) or [📌 uncertainty](https://csinva.io/notes/research_ovws/ovw_transfer_learning.html).* Most of these approaches can be combined with metalearning.

- finetuning
  - finetune all DNN params
  - finetune linear layer on activations
    - standard - train linear model on the embedding of the first token (usually an added `[CLS]` token) [peters et al. 2018](https://aclanthology.org/N18-1202/)
    - finetune linear model on all the activations
      - e.g. [evci, et al. 2022](https://arxiv.org/abs/2201.03529) - learn linear layer (using group-lasso) on features extracted from all layers
  - finetune specific DNN params (e.g. just the bias terms)
    - Cutting Down on Prompts and Parameters ([logan...sameer singh, riedel, 2021](https://arxiv.org/abs/2106.13353)) - finetune only the bias terms; works even with null prompts
- adapter - finetune lightweight layers on top of pre-trained layers (between finetuning all layers, and just finetuning a new layer)
  - add some new layers and retrain some specific things (all human choices)
  - side-tuning ([zhang, sax...malik, 2020](https://link.springer.com/chapter/10.1007/978-3-030-58580-8_41)) - train a “side” network that is fused with the pretrained model via summation
  - Combining Modular Skills in Multitask Learning ([ponti, sordoni, bengio, & reddy, 2022](https://arxiv.org/pdf/2202.13914.pdf)) - learn adaptor with disentangled inventory of skills
  - [Parameter-Efficient Transfer Learning for NLP](http://proceedings.mlr.press/v97/houlsby19a.html)
  - [AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/abs/2007.07779)
- predict a mask
  - ablate some model weights by training a binary mask over model parameters (Zhao et al., 2020; Radiya-Dixit and Wang, 2020)
  - predict mask over attention heads
- prompting = few-shot learning = priming = in-context learning (starts with GPT)
  - prompting without changing any model parameters
    - limitation: can't exploit sets longer than the training window
  - [MetaICL: Learning to Learn In Context](https://arxiv.org/abs/2110.15943) (min et al. 2022) - tune LLM to do in-context learning on a large set of training tasks (few-show prompting and training time and at test-time)
  - [Visual Prompting via Image Inpainting](https://arxiv.org/abs/2209.00647) (bar...darrell, globerson, efros, 2022)
  - PatternExploiting Training (PET) -- Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference ([schick & schutze, 2021](https://aclanthology.org/2021.eacl-main.20.pdf))
    - **cloze questions** - same as masked language modeling: task is to replace some missing words
    - use cloze-question templates (e.g. it was "good" or "bad") to get soft labels for unlabeled data and then finetune on theses
- prompt-tuning (also see next section on autoprompting)
  - [Attentional Mixtures of Soft Prompt Tuning for Parameter-efficient Multi-task Knowledge Sharing](https://arxiv.org/abs/2205.11961)
  - [STT: Soft Template Tuning for Few-Shot Adaptation](https://arxiv.org/abs/2207.08408)
- misc....(few shot mostly)
  - LM-BFF [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/abs/2012.15723) (gao et al. 2020)
    - uses T5 to generate (i) template for the task (which might include a whole example or two) + (i) appropropriate label tokens in the vocabulary for the task (suffers from computationally intensive search + sub-optimal discrete space search)
  - [Adapting Language Models for Zero-shot Learning by Meta-tuning on Dataset and Prompt Collections](https://arxiv.org/abs/2104.04670) (zhong...dan klein, 2021)

**mt-dnn line of work**

- Multi-Task Deep Neural Networks for Natural Language Understanding ([xiaodong liu ... gao 2019](https://aclweb.org/anthology/papers/P/P19/P19-1441/)) - multi-task learning on the 9 glue tasks (first layers are shared, then some task-specific layers at top)
- RAdam: On the Variance of the Adaptive Learning Rate and Beyond ([liyuan liu...gao, han, 2020](https://openreview.net/pdf?id=rkgz2aEKDr))
  - usually need to do learning-rate warmup when trainin (e.g. with Adam)
  - RAdam = add a term to rectify the variance of the adaptive learning rate in Adam

- SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization ([jiang...gao, zhao, 2020](https://aclanthology.org/2020.acl-main.197/))
  1. Smoothness-inducing regularization, which effectively manages the complexity of the model
  2. Bregman proximal point optimization to prevent aggressive updating

- The Microsoft Toolkit of Multi-Task Deep Neural Networks for Natural Language Understanding ([xiaodong liu...gao, 2020](https://aclanthology.org/2020.acl-demos.16/))
- Posterior Differential Regularization with f-divergence for Improving Model Robustness ([hao cheng, ..., gao 2021](https://aclanthology.org/2021.naacl-main.85/))
  - regularize model posterior difference between clean + noisy inputs (e.g. adversarially attacked inputs)



# prompting

- Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing ([liu...neubig, 2021](https://arxiv.org/pdf/2107.13586.pdf))
  - from *feature-engineering* -> *architecture engineering* -> *prompt engineering*
  - ![prompting_typology](../assets/prompting_typology.png)
  
- LAMA [Language Models as Knowledge Bases?](https://arxiv.org/abs/1909.01066) (petroni...riedel, 2019) - Proposes using fill-in-the-blank (cloze) prompts for extracting knowledge from large language models
  - create LAMA probe - dataset of (subject, relation, object) triplets with templates -- find that BERT can recall these relations
  - [How to Query Language Models?](https://arxiv.org/abs/2108.01928) (adolphs et al. 2021) - query LLMs by example (e.g. "Ronaldo plays for Portugal. Who does Neuer play for?")
  - [How Can We Know What Language Models Know?](https://arxiv.org/abs/1911.12543) (jiang ... neubig, 2020)
    - mining-based and paraphrasing-based methods to automatically generate high-quality diverse prompts
    - ensemble methods to combine answers from different prompts (e.g. avg logits and more)
  - Noisy Channel Language Model Prompting for Few-Shot Text Classification ([min et al. 2022](https://arxiv.org/pdf/2108.04106.pdf))
    - Querying $P(question|answer)$ with Bayes rule outperforms standard querying $P(answer|question)$

## autoprompting

- [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://aclanthology.org/2020.emnlp-main.346/) (shin...sameer singh, 2020)
  - select prompts from a fixed set of tokens (resulting prompts are not coherent)
  - only work on MLM
  - elicit sentiment / factual knowledge
  - [Universal Adversarial Triggers for Attacking and Analyzing NLP](https://arxiv.org/abs/1908.07125) (wallace...sameer singh, 2019) - find input-agnostic sequences of tokens that trigger a model to produce a specific prediction when concatenated to any input from a dataset
  
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
  - [Do Prompt-Based Models Really Understand the Meaning of their Prompts?](https://arxiv.org/abs/2109.01247) (webson & pavlick, 2022) - models can learn fine with prompts that are intentionally irrelevant

## llm chaining

**many notes are from this [thread](https://twitter.com/iraphas13/status/1551959289023016967) on chaining models together**

- steering
  - overviews
    - [AI Chains: Transparent and Controllable Human-AI Interaction by Chaining Large Language Model Prompts](https://arxiv.org/abs/2110.01691) (wu et al. 2022) - chaining LLM steps together: output of one step becomes the input for the next
      - interactive system where users can modify chains + their intermediate results
    - [Language Model Cascades](https://arxiv.org/abs/2207.10342) (dohan...sutton, 2022) - treat chaining models as probabilistic programs
      - use a probabilistic-programming language (PPL) to define a joint probability model on string-valued random variables, parameterized using LMs, and then condition this model on string-valued observations in order to compute a posterior over string-valued unknowns
      - self-PPLs extend probabilistic graphical models to support more complex joint distributions whose size and “shape” can itself be stochastic
        - e.g., a graph unrolled for a random number of iterations, until a data-dependent stopping criterion is met
        - variables are all text: questions $Q$, answers $A$, and intermediate thoughts $T$
  - posthoc
    - Chain of Thought Prompting ([wei et al. 2022](https://arxiv.org/abs/2201.11903))
      - in few-shot prompts, don't just provide answer but also reasoning
      - model output then provides reasoning + answer
      - Self-Consistency Improves Chain of Thought Reasoning in Language Models ([wang, wei, schuurmans, quoc le, ... zhou, 2022](https://arxiv.org/abs/2203.11171)) - sample a diverse set of reasoning paths from a language model via chain of thought prompting then return the most consistent final answer in the set
    - scratchpads [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114) (nye et al. 2021)
    - selection inference ([creswell et al. 2022](https://arxiv.org/abs/2205.09712)) - generate set of facts, then iteratively generate inferences from the facts to yield the final answer
    - least-to-most prompting ([zhou...quoc le et al. 2022](https://arxiv.org/abs/2205.10625)) - prompt LLM with context showing how to reduce into subproblems; then LLM sequentially solves the subproblems, using the previous answers
    - Generated Knowledge Prompting for Commonsense Reasoning ([liu...hasjishirzi, 2021](https://arxiv.org/abs/2110.08387)) - generate knowledge from an LLM then prvide it as additional input when answering a question
  - training
    - verifiers ([cobbe et al. 2021](https://arxiv.org/abs/2110.14168)) - train model to judge whether an answer and thought are likely to be “valid”
    - maieutic prompting ([jung et al. 2022](https://arxiv.org/abs/2205.11822)) - generate a tree of all explanation of the form "True, because...", "False, because..." then query LLM with these as prompts
      - then use Max-SAT to try to satisfy as many relations between the model explanations as possible to come up with the true answer
    - subgoal search ([czechowski et al. 2021](https://t.co/PCR4yexHti)) - train model to generate subgoals then solve them in a graph
    - STaR “Self-taught reasoner” ([zelikman...goodman, 2022](https://arxiv.org/abs/2203.14465))
      - first, finetune on observed $(Q, T, A)$ triplets
      - then, impute unknown $T_i$ given dataset of pairs $(Q_i, A_i)$ by sampling until finding a $T_i$ which leads to the correct answer
  - robotics-specific
    - zero-shot planning ([huang, abbeel, pathak, & mordatch, 2022](https://arxiv.org/abs/2201.07207))
    - socratic models [arxiv.org/abs/2204.00598](https://arxiv.org/abs/2204.00598)
    - Inner Monologue [arxiv.org/abs/2207.05608](https://arxiv.org/abs/2207.05608)
    - [global workspace](https://arxiv.org/abs/2103.01197)
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
    - RETRO ([borgeaud et al. 2022](https://arxiv.org/abs/2112.04426)) - nearest neighbors to model's input are retrieved, encoded, and conditioned on with chunked cross-attention 
    - memorizing transformers ([wu...szegedy, 2022](https://arxiv.org/abs/2203.08913)) - knn-based learned indexing + retrieval at training time.
      - at test time, you just need to index the entire context and the model will be able to use it

# misc

## transformer circuits / reverse-engineering / editing

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
- Locating and Editing Factual Associations in GPT ([meng et al. 2022](https://arxiv.org/abs/2202.05262) )
  - *localize factual associations* - causal intervention for identifying neuron activations that are decisive in a model’s factual predictions
    - "causal traces" - run net multiple times, introducing corroptuions and then restoring states from original non-corrupted forward pass to see which states can restore the original results
    - a small number of states contain info that can flip the model from one state to another
  - *change factual associations* - modify feedforward weights to update specific factual associations using Rank-One Model Editing (ROME)
- Knowledge Neurons in Pretrained Transformers ([dai et al. 2021](https://arxiv.org/abs/2104.08696)) - integrated gradients wrt to each neuron in BERT

## symbolic reasoning

*See also notes on [📌 comp neuro](https://csinva.io/notes/research_ovws/ovw_comp_neuro.html).*

- GPT-3 [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) - simply adding “Let’s think step by step” before each answer increases the accuracy on MultiArith from 17.7% to 78.7% and GSM8K from 10.4% to 40.7% with GPT-3
- [Compositional processing emerges in neural networks solving math problems](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8491571/) (russin, roland fernandez, ..., smolensky, gao, 2021)
- neurocompositional computing ([smolensky…gao, 2022](https://arxiv.org/abs/2205.01128))
  - longer tutorial ([smolensky, …, gao, 2022](https://www.microsoft.com/en-us/research/uploads/prod/2022/04/Neurocompositional_computing__tutorial.pdf))
  
  - *central paradox of cognition* is that brain both uses continuous neural symbols but is compositional ([smolensky et al. 1992](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.20.4352&rep=rep1&type=pdf))
    - Compositionality
    - Continuity - the encoding and processing of information is formalized with real numbers that vary continuously
  
  - 3 challenges
    - compositional generalization
    - data efficiency
    - comprehensibility
  - solution - NECST: Neurally-Encoded Compositionally-Structured Tensor computing ([smolensky & legendre, 2006](https://psycnet.apa.org/record/2006-07970-000)) - basically leverages TPR
    - TPR roles and fillers can both be made continuous
  
  - neural space vs symbolic space (many different things (e.g. sentences) can mean the same thing)
    - word vectors can be thought of as “soft symbols”
  - want to move from symbolic repr. to neural repr. while keeping interpretability
    - system should output intermediate steps in addition to answer
    - thinking fast (system 1: fast, intuitive) + slow (system 2: slower, logical, derivative)
  - concrete proposals
    - transformer activation vector should encode graph of flow through the network
      - ex. task: regurgitate a sequence
  
- TPR: Tensor product variable binding and the representation of symbolic structures in connectionist systems ([paul smolensky, 1990](https://www.sciencedirect.com/science/article/abs/pii/000437029090007M?via%3Dihub)) - activation patterns are "symbols" and internal structure allows them to be processed like symbols
  - tensor product representation = TPR
  - [TPR slides](https://www.mit.edu/~jda/teaching/6.884/slides/oct_02.pdf)
  - TPR of a structure is the sum of the TPR of its constituents
    - tensor product operation allows constituents to be uniquely identified, even after the sum (if roles are linearly independent)
  - **filler** - one vector that embeds the content of the constituent
  - **role** - second vector that embeds the structural role it fills
-  NECSTransformer: [Enhancing the Transformer with Explicit Relational Encoding for Math Problem Solving](https://www.microsoft.com/en-us/research/publication/enhancing-the-transformer-with-explicit-relational-encoding-for-math-problem-solving/) (schlag, ..., gao, 2019)
  - TP-attention
  - beat SOAon free-form math word-problems
  - in addition to K, Q, V, also add a role-vector
    - do element-wise multiplication of outputted vector with role-vector
  - TPR built as tensor product of 2 vectors:
    - filler - the vector returned by attention
      - ex. one head learns "second-argument-of"
    - role - a relation conceptually labeling an edge of the attention graph
- [TP-N2F: Tensor Product Representation for Natural To Formal Language Generation - Microsoft Research](https://www.microsoft.com/en-us/research/publication/natural-to-formal-language-generation-using-tensor-product-representations/) (chen...gao, 2019)

## neuro-inspired

- recurrent sparse reconstruction ([shi...joshi, darrel, wang, 2022](https://arxiv.org/pdf/2204.10962.pdf)) - sparse reconstruction (of a single image) learns a layer that does better than self-attention

## sparse experts / ensembles / mixture of experts (MoE)

- note: nowadays often the "experts" are different MLPs following the self-attention layers
- A Review of Sparse Expert Models in Deep Learning ([fedus, jeff dean, zoph, 2022](https://arxiv.org/abs/2209.01667))
  - sparsity decouples the parameter count from the compute per example allowing for extremely large, but efficient models
  - routing algorithm - determines where to send examples
    - discreteness makes it difficult
      - some works use RL to learn routing
      - standard approach uses gumbel-softmax
      - usually get matrix of similarities between input tokens and experts and route based on these
        - sometimes route to topk experts rather than top1
    - load balancing - usually add an auxiliary loss to encourage equal tokens being sent to different experts
- non-specialized experts
  - Early versions ([Jacobs, michael jordan, nowlan, & hinton, 1991](https://ieeexplore.ieee.org/abstract/document/6797059)) had independent feed-forward networks serving as experts
  - Sparsely-gated MOE layer ([Shazeer...quoc le, hinton, dean, 2017](https://arxiv.org/abs/1701.06538)) have been studied with token-based routing with backprop
  - replace FFN in transformers with expert layers
    - GShard [Lepikhin et al. (2021)](https://arxiv.org/abs/2006.16668), which appplies this concept to machine translation
    - Switch transformers [Fedus et al. (2022)](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf) simplifies the architecture to activation of only one expert per layer
  - BASE Layers [Lewis et al. (2021)](https://proceedings.mlr.press/v139/lewis21a.html) - find an alternative approach to routing by formulating it as a linear assignment problem
  - Hash layers [Roller et al. (2021)](https://arxiv.org/abs/2106.04426) use a fixed hash as the gating function
- specialized experts as fully independent models (sometimes for multi-task learning)
  - DEmix Layers [Gururangan et al.](https://arxiv.org/abs/2108.05036) (2022) --  DEMix layers – placed in the feedforward layers of the Transformer – contain experts which specialize on specific domains. Routing at train time is determined only by the domain label, but all experts are activated at inference time and mixed according to weights estimated from a validation set
  - [Sparsely Activated Mixture-of-Experts are Robust Multi-Task Learners](https://arxiv.org/abs/2204.07689) (gupta...awadallah, gao, 2022) - use task description to improve routing
  - [Pfeiffer et al. (2022)](https://arxiv.org/abs/2205.06266) - multilingual expert model with language-specific routing
  - task-level MoE [Kudugunta et al. (2021](https://arxiv.org/abs/2110.03742)) -- multi-task expert model with task-specific routing
  - ELMS -- Branch-Train-Merge ([li et al. 2022](https://arxiv.org/abs/2208.03306))
    - parallel language model of smaller expert LMs
    - each can be added/removed, ensembled, or parameter-averaged at any time for efficient scaling and rapid customization
    - improves perplexities, when controlling for training cost
      - require expert domain specialization
- ensembles (some of these are non-transformer papers)
  - model soups ([wortsman...schmidt, 20221](https://proceedings.mlr.press/v162/wortsman22a.html)) - average weights of finetuned models
    - snapshot ensembles - average different checkpoints during training ([huang et al. 2017](https://arxiv.org/abs/1704.00109))
    - stochastic weight averaging ([izmailov, ..., wilson, 2019](https://arxiv.org/abs/1803.05407v3)) - average multiple checkpoints during training
    - batch ensemble ([wen et al. 2020](https://arxiv.org/pdf/2002.06715.pdf)) - have several rank-1 keys that index different weights hidden within one neural net
  - fit many models into one
    - superposition of many models into one ([cheung...olshausen, 2019](https://proceedings.neurips.cc/paper/2019/hash/4c7a167bb329bd92580a99ce422d6fa6-Abstract.html)) - both during training/testing models are indexed via a high-dim key for each task
    - supermasks in superposition ([wortsman, ..., yosinski, farhadi, 2020](https://proceedings.neurips.cc/paper/2020/hash/ad1f8bb9b51f023cdc80cf94bb615aa9-Abstract.html)) - randomly fixed based net + for each task finds subnet that chieves good performance
      - if task identity not given, correct subnet inferred by minimizing output entropy
    - Git Re-Basin: Merging Models modulo Permutation Symmetries ([ainsworth, hayase, & srinivasa, 2022](https://arxiv.org/abs/2209.04836)) - algo to merge models even when they haven't been pretrained together

## causal inference / llm querying

- [InferBERT: A Transformer-Based Causal Inference Framework for Enhancing Pharmacovigilance](https://www.frontiersin.org/articles/10.3389/frai.2021.659622/full) (wang...liu, 2021) - learn + test feature relationships from attention weights
- [CausaLM: Causal Model Explanation Through Counterfactual Language Models | Computational Linguistics](https://direct.mit.edu/coli/article/47/2/333/98518/CausaLM-Causal-Model-Explanation-Through) (2021) - produce example-level causal model explanations using models finetuned on auxiliary adversarial tasks derived from the causal graph of the problem
- [Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, Stuart Shieber. Investigating Gender Bias in Language Models Using Causal Mediation Analysis. NeurIPS 2020.](https://proceedings.neurips.cc/paper/2020/file/92650b2e92217715fe312e6fa7b90d82-Paper.pdf)
  - Applies causal mediation analysis to identify decisive neurons and attention heads responsible for gender bias in large language models
  - Identifies a small handful of decisive attention heads in this case
- [Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg. Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. TACL 2021.](https://arxiv.org/pdf/2006.00995.pdf) - Proposes measuring the importance of specific information within a model by introducing a causal intervention to erase that information, then observing the causal effects.
- nucleus sampling: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (holtzman...choi, 2019)
- applications
  - [AI-based language models powering drug discovery and development - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1359644621002816) (liu et al. 2021)

## open issues

- [Shortcut Learning of Large Language Models in Natural Language Understanding: A Survey](https://arxiv.org/abs/2208.11857) (du et al. 2022)

## cool tasks

- [Forecasting Future World Events with Neural Networks](https://arxiv.org/abs/2206.15474) (zou...hendrycks, 2022)

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

## huggingface tutorial

Broadly, models can be grouped into three categories:

- GPT-like (also called *auto-regressive* Transformer models)
- BERT-like (also called *auto-encoding* Transformer models)
- BART/T5-like (also called *sequence-to-sequence* Transformer models)
- [Handling multiple sequences - Hugging Face Course](https://huggingface.co/course/chapter2/5?fw=pt)
  - pad sequences to have the same length (need to modify attention masks to ignore the padded values)