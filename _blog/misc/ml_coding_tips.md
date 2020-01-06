---
layout: notes_without_title
section-type: notes
title: ml coding tips
category: blog
---

# Useful ml / data-science packages and tips

**chandan singh** 

---

**These are some packages/tips for machine-learning coding in python.**

Machine learning code gets messy fast. In contrast to other tasks, machine learning often require a large number of variations of very similar models that can often be difficult to keep track of. Here are some tips on coding for machine learning, assuming you already know the basics (e.g. [numpy](http://www.numpy.org/), [scikit-learn](https://scikit-learn.org/stable/), etc.) and have selected an appropriate framework for your problem (e.g. [pytorch](https://pytorch.org/)):

### very useful packages

- [tqdm](https://github.com/tqdm/tqdm): add a loading bar to any loop in a super easy way:

```python
for i in tqdm(range(10000)):
	...
```
displays 
```python
76%|████████████████████████████         | 7568/10000 [00:33<00:10, 229.00it/s]
```

- [h5py](http://docs.h5py.org/en/stable/): a great way to read/write to arrays which are too big to store in memory, as if they were in memory
- [slurmpy](https://github.com/brentp/slurmpy): lets you submit jobs to slurm using just python, so you never need to write bash scripts.
- [pandas](https://pandas.pydata.org/): provides dataframes to python - often overlooked for big data which might not fit into DataFrames that fit into memory. Still very useful for comparing results of models, particularly with many hyperparameters.
  - [modin](https://github.com/modin-project/modin) - drop in pandas replacement to speed up operations
- [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling) - very quick data overviews
- [pandas bamboolib](https://www.youtube.com/watch?v=yM-j5bY6cHw&feature=youtu.be) - ui for pandas, in development
- [dovpanda](https://github.com/dovpanda-dev/dovpanda) - helpful hints for using pandas
- [tabnine](https://github.com/wenmin-wu/jupyter-tabnine?utm_source=share&utm_medium=ios_app&utm_name=iossmf) - autocomplete for jupyter
- cloud9sdk can be a useful subsitute for jupyterhub



### computation

- [dask](https://dask.org/) - natively scales python
- [jax](https://github.com/google/jax) - high-performance python + numpy
- [numba](https://numba.pydata.org/) - alternative to dask, just requires adding decorators to functions
- [some tips for using jupyter](https://github.com/NirantK/best-of-jupyter)
  - useful shortcute `tab`, `shift+tab`: inspect something



### plotting

- [matplotlib](https://matplotlib.org/) - basic plotting in python
- [animatplot](https://github.com/t-makaro/animatplot) - animates plots in matplotlib
- [seaborn](https://seaborn.pydata.org/) - makes quick and beautiful plots for easy data exploration, although may not be best for final plots
- [bokeh](https://bokeh.pydata.org/en/latest/) - interactive visualization library, [examples](https://github.com/WillKoehrsen/Bokeh-Python-Visualization)  
- [plotly](https://plot.ly/python/offline/) - make interactive plots


### general tips

- **installing things**: using pip/conda is generally the best way to install things. If you're running into permission errors `pip install --user` tends to fix a lot of common problems
- **make classes for datasets/dataloaders**: wrapping data loading/preprocessing allows your code to be much cleaner and more modular. It also lets your models easily be adapted to different datasets. Pytorch has a good [tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) on how to do this (although the same principles apply without using pytorch.
- **store hyperparameters**: when you test many different sets of hyperparameters, it is difficult to easily map which hyperparameters correspond to which results. It's important to store hyperparameters in a easily readable way, such as saving an [argparse object](https://docs.python.org/3/library/argparse.html), or storing/saving parameters in a class you define yourself.


### environment

- it's hard to pick a good ide for data science. [jupyter](https://jupyter.org/) notebooks are great for exploratory analysis, while more fully built ides like [pycharm](https://www.jetbrains.com/pycharm/) are better for large-scale projects
- using [atom](https://atom.io/) with the [hydrogen](https://atom.io/packages/hydrogen) plugin often strikes a nice balance
- [jupytertext](https://github.com/mwouts/jupytext) offers a nice way to use version control with jupyter
- [pdoc3](https://pypi.org/project/pdoc3/) can very quickly generate simple api from docstrings

### hyperparameter tracking

- it can often be messy to keep track of ml experiments
- often I like to create a class which is basically a dictionary for params I want to vary / save and then save those to a dict (ex [here](https://github.com/csinva/dnn-experiments/tree/master/vision_fit)), but this only works for relatively small projects
- [trains](https://github.com/allegroai/trains) by allegroai seems to be a promising experiment manager
- [reddit thread](https://www.reddit.com/r/MachineLearning/comments/bx0apm/d_how_do_you_manage_your_machine_learning/) detailing different tracking frameworks

### command line utils

- `ctrl-a`: HOME, `ctrl-e`: END
- `git add folder/\*.py`

### sharing

- [gradio](https://github.com/gradio-app/gradio) yields nice web interface for getting model predictions

### cheat-sheets

- [dalex](https://github.com/pbiecek/DALEX) 
- [microsoft-interpret](https://github.com/microsoft/interpret)

### converting between dl frameworks

- [mmdnn](https://github.com/microsoft/MMdnn)

### vim shortcuts

- remap esc -> jk
- J - remove whitespace
- ctrl+r - redo
- o - new empty line below (in insert mode)
- O - new empty line above (in insert mode)
- e - move to end of word
- use h and l!
- r - replace command
- R - enter replace mode (deletes as it inserts)
- c - change operator (basically delete and then insert)
  - use with position (ex. ce)
- W, B, gE, E - move by white-space separated words
- ctrl+o - go back, end search
- % - matching parenthesis
  - can find and substitute
- vim extensive
  - ! ls

### misc services

- google buckets: just type gsutil and then can type a linux command (e.g. ls, du)

### tmux shortcuts

- remap so clicking/scrolling works
- use ctrl+b to do things
- tmux ls

### installation

- pip use `--user`

### sharing

- [bookdown](https://bookdown.org/) - write books in markdown

### reference

- [this repo](https://raw.githubusercontent.com/r0f1/datascience/master/README.md) and [this repo](https://github.com/r0f1/datascience) has a more comprehensive and useful list
- feel free to use/share this openly
- for similar projects, see some other repos: (e.g. [acd](https://github.com/csinva/acd)) or my website ([csinva.github.io](https://csinva.github.io/))
