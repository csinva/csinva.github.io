---
layout: notes
title: reproducibility
category: cs
---

{:toc}

Some notes on different programming techniques / frameworks for reproducibility

# containerization

- e.g. [docker](https://docs.docker.com/get-started/overview/)



# data version control

- [dvc](https://dvc.org/) - data version control

  - .dvc folder keeps track of some internal stuff (like .git)
  - metafiles ending with .dvc are stored in git, tracking big things like data and models

  - also simple support for keeping track of metrics, displaying pipeline, making plots

  - keep track of old things using git checkout and dvc checkout	

  - [dagshub](https://dagshub.com/docs/experiment-tutorial/overview/) - built on dvc, like github (gives ~10GB free storage per project)

    - “Our recommendation is to separate distinct experiments (for example, different types of models) into separate branches, while smaller changes between runs (for example, changing model parameters) are consecutive commits on the same branch.”
     - not open source :frowning_face:
 - [replicate.ai](https://replicate.ai/) - version control for ml
   - lightweight, focuses on tracking model weights / sharing + dependencies
   - less about hyperparams
 - [mlflow](https://mlflow.org/) (open-source) from databricks

   - API and UI for logging parameters, code versions, metrics and output files
 - [clear-ml](https://github.com/allegroai/trains)

 - [gigantum](https://www.youtube.com/watch?v=He0hBcq49Gw) - like a closed-source dagshub
 - [codalab](https://codalab-worksheets.readthedocs.io/en/latest/) - good framework for reproducibility
 - paid / closed-source
   - [weights and biases](https://www.wandb.com/) (free for academics, paid otherwise)
   - [neptune.ai](https://neptune.ai/)
    - [h20 ai](https://www.youtube.com/watch?time_continue=149&v=ZqCoFp3-rGc&feature=emb_logo) (source [here](https://github.com/h2oai/h2o-3))



# hyperparameter tuning

- weaker versions
   - [tensorboard](https://www.tensorflow.org/tensorboard) (mainly for deep learning)
- pytorch-lightning + hydra
- [ray](https://github.com/ray-project/ray)



# weights and biases

- `wandb.login()` - login to W&B at the start of your session
- `wandb.init()` - initialise a new W&B, returns a "run" object
- `wandb.log()` - logs whatever you'd like to log



# workflow management

- [prefect](https://www.prefect.io/core)
   - **tasks** are basically functions
   - **flows** are used to describe the dependencies between tasks, such as their order or how they pass data around
