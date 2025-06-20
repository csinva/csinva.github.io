---
layout: notes
title: Data science benchmarks for AI systems
category: blog
---

**ScienceAgentBench** ([chen...huan sun, 2024](https://arxiv.org/abs/2410.05080)) - 102 scientific coding tasks (from 44 papers in 4 disciplines validated by 9 subject-matter experts)

- target output for every task is a self-contained Python file
- each task has (a) task instruction, (b) dataset info, (c) expert-provided info and (d) a groundtruth annotated program
![Screenshot 2025-06-19 at 2.19.17 PM](../../_notes/assets/Screenshot%202025-06-19%20at%202.19.17%E2%80%AFPM.png)

<img src="{{ site.baseurl }}/notes/assets/Screenshot%202025-06-19%20at%202.19.17%E2%80%AFPM.png" class="noninverted medium_image"/>

**AutoSDT**: Scaling Data-Driven Discovery Tasks Toward Open Co-Scientists ([li...huan sun, 2025](https://arxiv.org/abs/2506.08140)) - 5k scientific coding tasks automatically scraped from github repos for papers (as a sanity check, they manually verified that a subset were reasonable)
![Screenshot 2025-06-19 at 2.22.52 PM](../../_notes/assets/Screenshot%202025-06-19%20at%202.22.52%E2%80%AFPM.png)

**DiscoveryBench**: Towards Data-Driven Discovery with Large Language Models ([majumder...clark, 2024](https://arxiv.org/abs/2407.01725)) - 264 tasks collected across 6 diverse domains, such as sociology and engineering, by manually deriving discovery workflows from papers

  - each task has datasets, metadata, natural-language discovery goal
![Screenshot 2025-06-19 at 2.18.31 PM](../../_notes/assets/Screenshot%202025-06-19%20at%202.18.31%E2%80%AFPM.png)

**BLADE**: Benchmarking Language Model Agents for Data-Driven Science ([gu...althoff, 2024](https://arxiv.org/pdf/2408.09667)) - 12 tasks, each has a (fairly open-ended) research question, dataset, and groundtruth expert-conducted analysis
![Screenshot 2025-06-19 at 4.22.04 PM](../../_notes/assets/Screenshot%202025-06-19%20at%204.22.04%E2%80%AFPM.png)

**Mlagentbench**: Benchmarking LLMs As AI Research Agents ([huang, vora, liang, & leskovec, 2023](https://arxiv.org/abs/2310.03302v2)) - 13 prediction tasks, e.g. CIFAR-10, BabyLM, kaggle (evaluate via test prediction perf.)

![Screenshot 2025-06-19 at 4.02.49 PM](../../_notes/assets/Screenshot%202025-06-19%20at%204.02.49%E2%80%AFPM.png)

**IDA-Bench**: Evaluating LLMs on Interactive Guided Data Analysis ([li...jordan, 2025](https://arxiv.org/pdf/2505.18223)) - scraped 25 notebooks from recent kaggle competitions, parse into goal + reference insights that incorporate domain knowledge

  - paper emphasizes interactive setting: evaluates by using the instruction materials to build a knowledgeable user simulator and then tests data science agents' ability to help the user simulator improve predictive performance
![Screenshot 2025-06-19 at 4.39.46 PM](../../_notes/assets/Screenshot%202025-06-19%20at%204.39.46%E2%80%AFPM.png)

**InfiAgent-DABench**: Evaluating Agents on Data Analysis Tasks ([hu...wu, 2024](https://arxiv.org/abs/2401.05507)) - 257 precise (relatively easy) questions that can be answered from 1 of 52 csv datasets
![Screenshot 2025-06-19 at 3.53.53 PM](../../_notes/assets/Screenshot%202025-06-19%20at%203.53.53%E2%80%AFPM.png)