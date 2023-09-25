---
layout: notes
title: Paper writing tips / nitpicks
category: blog
---

A (growing) list of nitpicks that may help improve a paper (mostly focused on ML conference papers). This list is all opinions, but for brevity I’ll state them as facts.

In general, I highly recommend the book [storytelling with data](https://github.com/Saurav6789/Books-/blob/master/storytelling-with-data-cole-nussbaumer-knaflic.pdf) (a very quick read despite being many pages).


### Figures

- For every figure, think carefully about the message you intend it to convey and how it could be misinterpreted
- Choose a single message to send with a figure and relentlessly remove all distractors
    - If a figure makes multiple points, split it into multiple figures
    - Clear figures ≠ beautiful figures
- Use colors for curves/bars very sparingly. For ML plots with many curves, there are usually only a couple things to highlight (highlight only a few curves with color and leave the rest gray, e.g. [this plot](https://github.com/csinva/data-viz-utils/blob/master/docs/img/conference_trends.png))
    - If curves have a natural ordering, then color them with a colormap, preferably a perceptually uniform one like viridis
    - Use diverging colormaps that are centered properly for values that are diverging like correlations (e.g. [this plot](https://seaborn.pydata.org/examples/many_pairwise_correlations.html))
    - Make figures colorblind-friendly (importantly, don’t use red & green together)
    - Keep colors consistent across different figures
- Remove everything unnecessary
    - Remove top / right splines
    - Remove gridlines unless the precise values on a graph are useful for a reader
- Label subpanels / individual points so they can be referenced directly in the text
- Fonts should be readable: generally very similar size to the baseline fontsize of the paper
- Figs should be in a vector format (e.g. pdf) so that their resolution stays sharp even when zoomed in. The only exception to this is when you’re plotting a *ton* of data, vector files can become very large and you should instead use a rasterized format with high resolution (e.g. png at 300 dpi).

### Tables

- Right-align numbers and keep rounding consistent (this makes it so that the decimal point lines up regardless of how large the numbers are)
- Use bolding for emphasis
- Show averages over different elements when possible

### Writing

- Write in present tense
- First-person is fine when it simplifies writing
- Make your abstract really good: most people will only ever read the abstract
    - I prefer longer abstracts that give a description of the results
- Introduction
    - AI papers often start with a grand opening
        - This is okay, but make the scope and limitations of your paper clear in the intro
    - Usually, the intro should have a compelling, easy-to-digest figure that motivates your approach; focus here on motivation rather than method details
- Methods
    - Remove all jargon and math that is not strictly needed for your method
    - Give your method a catchy name that is easy to remember and reference (ChatGPT is pretty good at this; [this website](https://csinva.io/acronym-generator/) may also help)
- Results
    - Start the results with a single figure/table with your key finding
    - Show ablation results with respect to each important hyperparameter/setting separately
    - Figure/table captions should only describe the data; main text should contextualize the results for the paper
- Discussion
    - Don’t just recap results
    - Address limitations and directions for future work

### General tips

- Start with a paper outline and iteratively refine it to fill in details
    - This makes it easier to elicit feedback
    - Save the final polishing of writing / cleaning figures until everything is set
- Writing should be clear and easy to skim

<br/>