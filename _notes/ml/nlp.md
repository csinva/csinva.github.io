---
layout: notes
title: nlp
category: ml
typora-copy-images-to: ../assets
---

{:toc}

Some notes on natural language processing, focused on modern improvements based on deep learning.

# nlp basics

*basics come from book "Speech and Language Processing"*

- **language models** - assign probabilities to sequences of words
  - ex. **n-gram model** - assigns probs to short sequences of words, known as n-grams
    - for full sentence, use markov assumption
  - eval: **perplexity (PP)** - inverse probability of the test set, normalized by the number of words (want to minimize it)
    - $PP(W_{test}) = P(w_1, ..., w_N)^{-1/N}$
    - can think of this as the weighted average branching factor of a language
    - should only be compared across models w/ same vocab
  - vocabulary
    - sometimes closed, otherwise have unkown words, which we assign its own symbol
    - can fix training vocab, or just choose the top words and have the rest be unkown
- **topic models (e.g. LDA)** - apply unsupervised learning on large sets of text to learn sets of associated words
- **embeddings** - vectors for representing words
  - ex. **tf-idf** - defined as counts of nearby words (big + sparse)
    - TF * IDF = [ (Number of times term t appears in a document) / (Total number of terms in the document) ] * log(Total number of documents / Number of documents with term t in it).
    - pointwise mutual info - instead of counts, consider whether 2 words co-occur more than we would have expected by chance
  - ex. **word2vec** - short, dense vectors
    - intuition: train classifier on binary prediction: is word $w$ likely to show up near this word? (algorithm also called skip-gram)
      - the weights are the embeddings
    - also **GloVe**, which is based on ratios of word co-occurrence probs
- some tasks
  - tokenization
  - pos tagging
  - named entity recognition
    - nested entity recognition - not just names (but also Jacob's brother type entity)
  - sentiment classification
  - language modeling (i.e. text generation)
  - machine translation
  - hardest: coreference resolution
  - question answering
  - [natural language inference](https://www.aclweb.org/anthology/P19-1334.pdf) - does one sentence entail another?
- most popular datasets
  - (by far) WSJ
  - then twitter
  - then Wikipedia
- [eli5](https://eli5.readthedocs.io/en/latest/libraries/sklearn.html#library-scikit-learn) has nice text highlighting for interp

# dl for nlp

- some recent topics based on [this blog](http://jalammar.github.io/)
- rnns
  - when training rnn, accumulate gradients over sequence and then update all at once
  - **stacked rnns** have outputs of rnns feed into another rnn
  - bidirectional rnn - one rnn left to right and another right to left (can concatenate, add, etc.)
- standard seq2seq
  - encoder reads input and outputs context vector (the hidden state)
  - decoder (rnn) takes this context vector and generates a sequence
- misc papers
  - [Deal or No Deal? End-to-End Learning for Negotiation Dialogues](https://arxiv.org/abs/1706.05125) (2017) - controversial FB paper where agents "make up their own language"



## attention / transformers

- self-attention layer [implementation](https://github.com/mertensu/transformer-tutorial) and [mathematics](https://homes.cs.washington.edu/~thickstn/docs/transformers.pdf)

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
- recent papers
  - [attention is all you need paper](<https://arxiv.org/abs/1706.03762>) - proposes transformer
  - [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) (by [Andrew Dai](https://twitter.com/iamandrewdai) and [Quoc Le](https://twitter.com/quocleix))
  -  [ELMo](https://arxiv.org/abs/1802.05365) (by [Matthew Peters](https://twitter.com/mattthemathman) and researchers from [AI2](https://allenai.org/) and [UW CSE](https://www.engr.washington.edu/about/bldgs/cse)) - no word embeddings - train embeddings w/ bidirectional lstm (on language modeling)
    - context vector is weighted sum of context vector at each word
  - [ULMFiT](https://arxiv.org/abs/1801.06146) ([Jeremy Howard](https://twitter.com/jeremyphoward) and [Sebastian Ruder](https://twitter.com/seb_ruder))
  - [OpenAI transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (by OpenAI researchers [Radford](https://twitter.com/alecrad), [Narasimhan](https://twitter.com/karthik_r_n), [Salimans](https://twitter.com/timsalimans), and [Sutskever](https://twitter.com/ilyasut))
  - [BERT](BERT) - semi-supervised learning (predict masked word - this is bidirectional) + supervised finetuning
  - [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (small released model, full trained model, even larger model from Nvidia)
  - [XLNet](https://arxiv.org/abs/1906.08237)
  - [roberta](https://arxiv.org/abs/1907.11692)
- these ideas are [starting to be applied to vision cnns](https://arxiv.org/abs/1904.09925)



# interpretable nlp

- [skip-gram model](https://arxiv.org/abs/1301.3781) (mikolov et al. 2013) - simplifies neural language models for efficient training of word embeddings
  - maximizing the probabilities of words being predicted by their context words
- [Neural Bag-of-Ngrams](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14513/14079) (li et al. 2017) - learn ngram repr. via deep version of skip-gram
- [fasttext](https://www.ijcai.org/Proceedings/16/Papers/401.pdf) (jin et al. 2016)
- [Improving N-gram Language Models with Pre-trained Deep Transformer](https://arxiv.org/abs/1911.10235) (wang et al. 2019) - use transformer to generate synthetic data for n-gram model

# huggingface tutorial

Broadly, models can be grouped into three categories:

- GPT-like (also called *auto-regressive* Transformer models)
- BERT-like (also called *auto-encoding* Transformer models)
- BART/T5-like (also called *sequence-to-sequence* Transformer models)

- [Tokenizers - Hugging Face Course](https://huggingface.co/course/chapter2/4?fw=pt)
  - word-based
    - punctuation splitting
    - need to do stemming (e.g. "dog" and "dogs")
    - unknown token [UNK] for anything not seen - to reduce the amount of this, can get character-based tokens
  - subword-based - break apart meaningful subparts of words
  - character-based - very little prior
  - many more (e.g. byte-level BPE, used in GPT-2)
- [Handling multiple sequences - Hugging Face Course](https://huggingface.co/course/chapter2/5?fw=pt)
  - pad sequences to have the same length (need to modify attention masks to ignore the padded values)