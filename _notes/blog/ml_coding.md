---
layout: notes
section-type: notes
title: writing good ml code
category: blog
---


# writing good ml code
**chandan singh**  
*last updated jul 10, 2018*

---


Machine learning code gets messy fast. In contrast to other programs, machine learning programs often require a large number of variations of similar models that can often be difficult to keep track of. Here are some tips on coding for machine learning:

1. Make a class for your dataset that includes funciton such as loading, preprocessing, and iterating through the data. This will allow you to easily do these things from different files, notebooks and be much cleaner
2. Store hyperparameters along with the results of your runs. I'd recommend saving hyperparameter sweeps into a pandas library.
3. The tqdm package displays a loading bar which is very useful