---
layout: notes
section-type: notes
title: nlp
category: ai
typora-copy-images-to: ./assets/nlp
---

# useful tools

- [eli5](https://eli5.readthedocs.io/en/latest/libraries/sklearn.html#library-scikit-learn) has nice text highlighting for interp

# nlp basics

- basics come from book "Speech and Language Processing"
- language models** - assign probabilities to sequences of words
  - ex. **n-gram model** - assigns probs to shorts squendes of words, known as n-grams
    - for full sentence, use markov assumption
  - eval: **perplexity (PP)** - inverse probability of the test set, normalized by the number of words (want to minimize it)
    - $PP(W_{test}) = P(w_1, ..., w_N)^{-1/N}$
    - can think of this as the weighted average branching factor of a language
    - should only be compared across models w/ same vocab
  - vocabulary
    - sometimes closed, otherwise have unkown words, which we assign its own symbol
    - can fix training vocab, or just choose the to V words and have the best be unkown
- **topic models (e.g. LDA)** - apply unsupervised learning on large sets of text to learn sets of associated words
- **embeddings** - vectors for representing words
  - ex. **tf-idf** - defined as counts of nearby words (big + sparse)
    - pointwise mutual info - instead of counts, consider whether 2 words co-occur more than we would have expected by chance
  - ex. **word2vec** - short, dense vectors
    - intuition: train classifier on binary prediction: is word w likely to show up near this word? (algorithm also called skip-gram)
    - the weights are the embeddings
    - also **GloVe**, which is based on ratios of word co-occurrence probs

# dl for nlp

- some recent topics based on [this blog](http://jalammar.github.io/)
- when training rnn, accumulate gradients over sequence and then update all at once
- **stacked rnns** have outputs of rnns feed into another rnn
- bidirectional rnn - one rnn left to right and another right to left (can concatenate, add, etc.)
- standard seq2seq
  - encoder reads input and outputs context vector (the hidden state)
  - decoder (rnn) takes this context vector and generates a sequence
- **attention**
  - encoder reads input and ouputs context vector after each word
  - decoder at each step uses a different weighted combination of these context vectors
    - specifically, at each step decoder concatenates its hidden state w/ the attention vector (the weighted combination of the context vectors)
    - this is fed to a feedforward net to output a word![Screen Shot 2019-04-11 at 7.57.14 PM](assets/nlp/Screen Shot 2019-04-11 at 7.57.14 PM.png)
- **transformer** - proposed in [attention is all you need paper](<https://arxiv.org/abs/1706.03762>)
  - self-attention - layer that lets word learn its relation to other layers
  - many stacked layers in encoder + decoder (not rnn - self-attention + feed forward)
  - details
    - initial encoding: each word -> vector
    - each layer takes a list of fixed size (hyperparameter e.g. length of longest sentence) and outputs a list of that same fixed size (so one output for each word)
      - can easily train with a masked word to predict the word at the predicted position in the encoding
  - self-attention 
    - for each word, want score telling how much importance to place on each other word (queries * keys)
      - softmax this and use it to do weighted sum of values![Screen Shot 2019-08-17 at 2.51.53 PM](assets/nlp/Screen Shot 2019-08-17 at 2.51.53 PM.png)
      - 
    - 3 components
      - queries
      - keys
      - values
  - multi-headed attention has several of each of these (then just concat them)
- recent papers
  - [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) (by [Andrew Dai](https://twitter.com/iamandrewdai) and [Quoc Le](https://twitter.com/quocleix))
  -  [ELMo](https://arxiv.org/abs/1802.05365) (by [Matthew Peters](https://twitter.com/mattthemathman) and researchers from [AI2](https://allenai.org/) and [UW CSE](https://www.engr.washington.edu/about/bldgs/cse)) - no word embeddings - train embeddings w/ bidirectional lstm (on language modelling)
    - context vector is weighted sum of context vector at each word
  - [ULMFiT](https://arxiv.org/abs/1801.06146) (by fast.ai founder [Jeremy Howard](https://twitter.com/jeremyphoward) and [Sebastian Ruder](https://twitter.com/seb_ruder)), the
  - [OpenAI transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (by OpenAI researchers [Radford](https://twitter.com/alecrad), [Narasimhan](https://twitter.com/karthik_r_n), [Salimans](https://twitter.com/timsalimans), and [Sutskever](https://twitter.com/ilyasut))
  - [BERT](BERT) - semi-supervised learning (predict masked word - this is bidirectional) + supervised finetuning
  - [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (small released model, full trained model, even larger model from Nvidia)
  - [XLNet](https://arxiv.org/abs/1906.08237)
  - [roberta](https://arxiv.org/abs/1907.11692)
- these ideas are [starting to be applied to vision cnns](https://arxiv.org/abs/1904.09925)