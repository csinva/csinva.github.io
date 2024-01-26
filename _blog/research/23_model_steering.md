---
layout: notes
title: Model steering
category: blog
---

*We taught models to write. Now can they follow instructions?*

The recent success of large language models (LLMs), such as ChatGPT, has ushered in a variety of new of NLP usecases. These usecases bring new challenges regarding *model steering* and *alignment* (see this [nice review](https://arxiv.org/abs/2309.15025)). In this post, I'll discuss some of these challenges, established methods for addressing them, and a couple directions I find interesting in this area.

## Definitions and existing methods

Model steering is broadly the process of controlling the output of a model. This is generally achieved through [prompting](https://en.wikipedia.org/wiki/Prompt_engineering), where a model is given a question in text and queried for an answer.

In this post, I'll share brief vignettes for a couple recent projects on LLM controllability: 

1. Tree prompting ([EMNLP 2023](https://arxiv.org/abs/2310.14034)) - simple method to boost performance by ensembling black-box LLM outputs with decision trees
2. Attention steering ([ICLR 2024](https://arxiv.org/abs/2311.02262)) - help users improve instruction-following for white-box LLMs by emphasizing parts of the prompt, e.g. the instruction
3. Augmented interpretable models ([Nature communications, 2023](https://arxiv.org/abs/2209.11799)) - use LLMs to help build fully interpretable, editable models (e.g. linear models or decision trees) that perform surprisingly well on text classification

## Improving prompting

<img src="assets/controllability.svg" class="full_image">

