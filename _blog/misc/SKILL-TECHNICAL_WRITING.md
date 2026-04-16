---
name: clear-technical-writing
description: Improve the clarity of technical writing and accompanying visualizations.
version: 1.0.0
license: MIT
compatibility: Any AI coding assistant that supports agentskills.io SKILL.md format (Claude Code, Cursor, VS Code Copilot, Hermes Agent, OpenHands, etc.) or OpenClaw. No external tools or APIs required.
metadata:
  author: Chandan Singh, some copying from https://github.com/conorbronsdon/avoid-ai-writing
  tags: writing editing quality
  agentskills_spec: "1.0"
---

# Clear technical writing

You are writing technical content that should be precise, clear, and grounded. Your goal is to convey a factual narrative, using the guidance below.

## Writing different sections

- Overall
  - Write in present tense
  - First-person is fine when it simplifies the writing
  - Text should move from simple to complicated, start with high-level motivation, then high-level solution before getting into detailed solution
  - Writing should be clear and easy to skim
  - Each paragraph should connect to the last. If paragraphs could be rearranged without the reader noticing, add connective tissue.
  - Each result should first be motivated and finally connected to the overall message of the report
- Abstract
  - Refine your abstract carefully before and after writing the rest of the paper
  - The abstract should always end with some description of a key result
- Introduction
  - Start with an overarching motivation for context, but make the scope and limitations of your reportclear in the intro
  - Usually, the intro should have a compelling, easy-to-digest figure that motivates your approach; focus here on motivation rather than method details
- Methods
  - Remove all jargon and math that is not strictly needed for your method
  - Give your method a name that is easy to remember and reference
- Results
  - Start the results with a single figure/table with your key finding
  - Show ablation results with respect to each important hyperparameter/setting separately
  - Figure/table captions should only describe the data; main text should contextualize the results for the paper
- Discussion
  - Don’t recap results
  - Address limitations and directions for future work

## Figures

- For every figure, think carefully about the message you intend it to convey and avoid any possible misinterpretations
- Choose a single message to send with a figure and relentlessly remove all distractors
  - If a figure makes multiple points, split it into multiple figures
- Use colors for curves/bars very sparingly. For plots with many curves, there are usually only a couple things to highlight and you can highlight them and leave the rest gray
  - If curves have a natural ordering, then color them with a colormap, preferably a perceptually uniform one like viridis
  - Use diverging colormaps that are centered properly for values that are diverging like correlations (e.g. [this plot](https://seaborn.pydata.org/examples/many_pairwise_correlations.html))
  - Make figures colorblind-friendly (importantly, don’t use red & green together)
  - Keep colors consistent across different figures
- Remove everything unnecessary
  - Remove top / right splines
  - Remove gridlines unless the precise values on a graph are useful for a reader
- Label subpanels / individual points so they can be referenced directly in the text
- All text in the figure should be readable and similar size
  - A good default is is the baseline fontsize of the text in the report
  - Avoid any overlapping text
- Figs should be in a vector format (e.g. pdf) so that their resolution stays sharp even when zoomed in.
  - The only exception to this is when you’re plotting a *ton* of data: in this case, vector files can become big and you should instead use a rasterized format with high resolution (e.g. png at 300 dpi)
- Do not invert axes, it is okay for lower to be "better"
- Legends should never cover data in the figure, often the best place for a legend is outside the figure (e.g. to the right or below)
- Figures should generally make good use of horizontal whitespace and avoid being too vertically long, which wastes space on a page
- Legend names and text annotations should be easily legible and avoid underscores, e.g. rename model_v1 to "Model (v1)"

## Tables

- Right-align numbers and keep rounding/padding consistent (this makes it so that the decimal point lines up regardless of how large the numbers are)
  - Numbers that begin with a decimal point should be left-padded with a "0"
- Use bolding for emphasis
- Whenever possible, include an "Average" column that show averages making the takeaway from a large table clear
- Show standard error in tables whenever possible, show it using $\pm$ notation and in a slightly smaller font than the main number

## Avoid AI writing

Avoid using the following:

- Em dashes (— and --), Bold overuse, Emojis, Excessive bullet lists
- The following phrases: delve / delve into, landscape (metaphor), tapestry, realm, paradigm, embark, beacon, testament to, robust, comprehensive, cutting-edge, leverage (verb), pivotal, underscores, meticulous / meticulously, seamless / seamlessly, game-changer / game-changing, hit differently / hits different, utilize, watershed moment, marking a pivotal moment, the future looks bright, only time will tell, nestled, vibrant, thriving, despite challenges… continues to thrive, showcasing, deep dive / dive into, unpack / unpacking, bustling, intricate / intricacies, complexities, ever-evolving, enduring, daunting, holistic / holistically, actionable, impactful, learnings, thought leader / thought leadership, best practices, at its core, synergy / synergies, interplay, in order to, due to the fact that, serves as, features (verb), boasts, presents (inflated), commence, ascertain, endeavor, keen (as intensifier), symphony (metaphor), embrace (metaphor)
- Default to "is" or "has" unless a more specific verb genuinely adds meaning
- Use sentence case for headings
- Too many headers in short text

## Writing process

- Start with a paper outline and iteratively refine it to fill in details
  - This makes it easier to elicit feedback
  - Save the final polishing of writing / cleaning figures until everything is set
- After finishing, go through the paper and make sure it is well-written, clear, and concise. Pretend you are a reviewer for NeurIPS and read through the paper critically, writing a detailed `self_review.md` report looking for any areas that could be improved or clarified, and suggesting any additional experiments or revisions. Then, make any necessary revisions to the paper based on your review, and write a final `response_to_self_review.md` document that addresses each point raised in the review, explaining how you have revised the paper in response.
- If you are editing a latex project, do not edit the `.bib` file or the `.sty` files, only the `.tex` files
  - You may write python code to help make figures or tables, which should be saved into their own subfolders, then loaded by the `.tex` files. Be thorough but concise in your writing, and make sure to clearly explain the methods and results.
  - Always use \cref for cross-referencing sections, figures, and tables in latex.
