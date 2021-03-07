---
layout: notes
title: presentations
---


Links to many of my presentations

### in progress
- [interpretation to causation](https://docs.google.com/presentation/d/1vwgpNp36ssspO5LcUgnohtm9JT-PeSwsX47LpZjvono/edit#slide=id.g9d8da641c4_0_36)
- [deep learning trends 2020](https://docs.google.com/presentation/d/15NRwGfKyGkcDIDKYwoq82PXdJrkJeRmSQfUbk5oQvuM/edit#slide=id.g8da3737a00_0_24)

### phd research overview
- [research overview](https://docs.google.com/presentation/d/19fTICv0pyRiwGE39mqE_eGTq0Arn3OtVnDSrURFbPMA/present?slide=id.p)
- [qual slides](https://docs.google.com/presentation/d/1cdzZsyRYRs9GiR9s2-V7OO8oIcaabT5TVJFGR9qk_HY/edit#slide=id.p)
    - [acd paige slides](https://docs.google.com/presentation/d/1aZvpZVk6pMmcUkQD0MaBQrmoecU835x6bqi1cbQNXeU/edit#slide=id.g51955b76d5_0_7)
        - Title: Disentangled interpretations for deep learning
        - Abstract: Deep learning models have achieved impressive predictive performance by learning complex functions of many variables, often at the cost of interpretability. I will discuss a recent line of work aiming to interpret neural networks by attributing importance to features and feature interactions for individual predictions. Importantly, the proposed methods disentangle the importance of features in isolation and the interactions between groups of features. These attributions significantly enhance interpretability and can be used to directly improve generalization in interesting ways. I will showcase how we have worked with domain experts to make these attributions useful in different computer vision settings, including in bioimaging and cosmology.
    - [dl joint reading group slides](https://docs.google.com/presentation/d/1MNEtfdD1ng8o_s75FYTZPbUDSmAf7smIc8jJnZ2FbO0/edit?usp=sharing)
        - Title: Disentangled interpretations for deep learning with ACD
        - Abstract: Deep learning models have achieved impressive predictive performance by learning complex functions of many variables, often at the cost of interpretability. I will discuss our recent works aiming to interpret neural networks by attributing importance to features and feature interactions for individual predictions. Importantly, the proposed method (named agglomerative contextual decomposition, or ACD) disentangles the importance of features in isolation and the interactions between groups of features. These attributions yield insights across domains, including in NLP/computer vision and can be used to directly improve generalization in interesting ways.
    - We focus on a problem in cosmology, where it is crucial to interpret how a model trained on simulations predicts fundamental cosmological parameters. By extending ACD to interpret transformations of input features, we vet the model by analyzing attributions in the frequency domain. Finally, we discuss ongoing work using ACD to develop simple transformations (e.g. adaptive wavelets) which can be both predictive and interpretable for cosmological parameter prediction.
        - Paper links: hierarchical interpretations [(ICLR 2019)](https://openreview.net/pdf?id=SkEqro0ctQ), interpreting transformations in cosmology [(ICLR workshop 2020)](https://arxiv.org/abs/2003.01926), penalizing explanations [(ICML 2020)](https://github.com/laura-rieger/deep-explanation-penalization)
    - [bair sem slides](https://docs.google.com/presentation/d/1vpunbuggj1sHxz3g_20pj2vtesHxJNUQoPbsHMMfz3A/edit#slide=id.p)
        - Title: Interpreting and Improving Neural Networks via Disentangled Attributions
        - Abstract: Machine learning models have achieved impressive predictive performance by learning complex functions of many variables. However, the inability to effectively interpret these functions has limited their use across many fields, such as science and medicine. This talk will cover a recent line of work aiming to interpret models by attributing importance to features / feature groups for a single prediction. Importantly, the proposed attributions disentangle the importance of features in isolation and the interactions between groups of features. These attributions are shown to yield insights across domains, including an NLP classification task and in understanding cosmological models. Moreover, these attributions can be used during training to improve generalization in interesting ways, such as forcing an image classification model to focus on shape instead of texture. The talk will place a large emphasis on rigorously evaluating and testing the proposed attributions to assure they are properly describing the model.
    - [acd + interpretable ml talk](https://docs.google.com/presentation/d/1x4zzugqu1kMhUKdM94MWrfB9Y6-YfrnHsv5MDFvzfmE/edit#slide=id.p)
        - [nlp version](https://docs.google.com/presentation/d/1bFIdoarqhZdwyXmvNKgqfbhKXByfgPW9h27HYGDXKGM/edit)
        - [interpretable ml](https://docs.google.com/presentation/d/13jbgFyYSSDaMUd2w4RY9GHteTcWJj1drS6_2sOkvnv4/present?slide=id.p) (discussion slides for group meeting)
    - [15 min overall talk](https://docs.google.com/presentation/d/16u_23-P7uvouNo_HvpLCgG1Um3ZDu5c3d76YBhQsjjI/edit#slide=id.gc5100c8fb5_0_519)
    - [cd + acd + cdep 5 min talk](https://docs.google.com/presentation/d/1qWor5d7AXVQfcLtQmFdMPUDPHeBkN1wWZid_pfxZjVQ/edit#slide=id.g51955b76d5_0_7)
    - [biohub meeting](https://docs.google.com/presentation/d/1-VadvMFR9UutmjGFXASMTdPLAiuU_6nzqKvPsEjKkW0/edit?usp=sharing) (w/ wooseok)
    - [dudoit group meeting](https://docs.google.com/presentation/d/1xA4aaBXKV4cO5_iXZZcUnheFpcMsT1U5fJZdaE7FwME/edit#slide=id.p)
    - [simons 2020 talk](https://docs.google.com/presentation/d/1OW4LxoKd6qxPaTP6XXH_gTLfrDV9DVfO7Wa00oznJv4/edit#slide=id.p) ([announcement](https://simons.berkeley.edu/events/disentangled-interpretations-and-how-we-can-use-them))
        - Title: Disentangled interpretations and how we can use them
        - Abstract: Recent machine learning models have achieved impressive predictive performance by learning complex functions of many variables, often at the cost of interpretability. This talk will cover recent work aiming to interpret models by attributing importance to features / feature groups for a single prediction. Importantly, the proposed attributions disentangle the importance of features in isolation and the interactions between groups of features. These attributions are shown to yield insights across domains, including an NLP classification task and in understanding cosmological models. Moreover, these attributions can be used during training to directly improve generalization in interesting ways. The talk will place a large emphasis on how to cogently define and evauate the proposed interpretations.

### dnn interp individual presentations

- [cogsci acd slides](https://docs.google.com/presentation/d/1I6djTqVn6YGKqxvQk59-4C39LbE68mNQbX1Go5pzTH4/present?slide=id.p)
    - [acd white-theme slides for bin](https://docs.google.com/presentation/d/1GjL0IJWft3IpdWxAwprXNiZkAG4Jl5mU6FNT2t0DGkQ/edit#slide=id.g5d6ee59cb4_0_0)
    - [acd bavrd](https://docs.google.com/presentation/d/1IVeb5ibe561VR5PAQEN5hdnwmpAHquK4JLb1vzT9fd8/present) (3 min pres. given at bavrd 2019)
- [cdep-focused slides 40 min](https://docs.google.com/presentation/d/1F9spZOwKbxtXqpCiKv2V332v4Vs-UV6mQ_7Xs_cnrLg/edit#slide=id.p)    
- [transform interp](https://docs.google.com/presentation/d/1mH1uG38qJg-ar0G-LiVPZWNKPO_2GiD-uayWM5AI-bo/present?slide=id.p)
- [pdr](https://docs.google.com/presentation/d/18Tdiym7hDeu0c4tj5XezrznIDfbltR4QQJJZjaHi7tk/present?slide=id.p)

### dnn misc individual presentations
- [faces final pres](https://docs.google.com/presentation/d/19Z4TnHCDkNENutyKmE_kZBSJX4jMUam6DoH3HckkMrI/present?slide=id.p)
    - [faces midterm pres](https://docs.google.com/presentation/d/1YxSZtsSOdQ_OZYgctFE1Cgfymv0cmBTzNDkt68d-RE0/present?slide=id.p)
    - [faces lit rvw](https://docs.google.com/presentation/d/1C6l4qq0O_-SosHswPwqo3ixJweMI8gkNku9crzZMw80/present?slide=id.p)
- [sensible local interpretations](https://docs.google.com/presentation/d/1tKVgg2bo7jSgE8TMyt15VAOkGhDPncFpMnZ-mmtoMNI/edit#slide=id.p) (5 min update for pacmed internship)
- [dnn experiments](http://csinva.github.io/pres/dnn_experiments/#/)

### teaching
- [machine learning (cs 189)](https://csinva.github.io/pres/189/#/)
- [intro to ai (cs 188)](https://csinva.github.io/pres/188/#/)
- [interpretability workshop](https://docs.google.com/presentation/d/1RIdbV279r20marRrN0b1bu2z9STkrivsMDa_Dauk8kE/present?slide=id.p)

### coursework
- [vissci ovw](https://docs.google.com/presentation/d/1d2prZlhmG72whCTzfGJ_pk-Pgv9Sx-bGG0e503whYvw/present?slide=id.p)
- [harms of ai](https://docs.google.com/presentation/d/1yZkEkDU-ELvh_Od3xvhd8Lsu70U7Cjs5QJnpNIVLgII/present) (1 hr talk given in possible minds course 2019)
- [hummingbird tracking](https://docs.google.com/presentation/d/15iygjXGLu7Ha096GwMV5t6sjP7IPcTHqimPkTDwrpPI/present?slide=id.p)

### undergrad
- [facebook pres](https://docs.google.com/presentation/d/1n_EImcRN8_R-smL6h-Gfxa11vsUczljj/present?slide=id.p1)
- [wsimule urds](https://docs.google.com/presentation/d/1GO6lN5o2idozOUdnObXGnXKFbZiJiKKKkmx73uE4BAI/present?slide=id.p) (5 min talk given at URDS 2017)
    - [wsimule tomtom](https://docs.google.com/presentation/d/1KghayB2g8u5xwVuILzT4XtalKi3rerVROaoRb6RUVBk/edit) (10 min talk given at Tom Tom 2017)
    - [wsimule slide](/assets/write_ups/wsimule_17_nips_slide.pdf) (1 min pres. given at AMLCID Neurips 2017 workshop)
- [linearization](https://docs.google.com/presentation/d/1JriXXofysuXyfU4CeyNHJUTYSfa18R9Q3EhkCwFwh4g/present?slide=id.p) (5 min talk given at URDS 2017)
- [sparse coding class pres](https://docs.google.com/presentation/d/199lCpVaOA6ez4QXkt9W8-fMv8rF_rGv_9rsXIwe1LKI/edit#slide=id.p)

### posters
- [acd poster](/assets/write_ups/acd_18_bairday_poster.pdf)
    - [acd + interp poster](/assets/write_ups/utokyo_19_interp_poster.pdf)
- [random forest image segmentation](/assets/write_ups/singh_15_rf_segmentation.pdf)
- [wsimule poster](/assets/write_ups/wsimule_17_nips_poster.pdf)

### recordings
- talks
    - [bair sem feb 2020 best vid](https://www.youtube.com/watch?v=avrnelFZSS4&feature=youtu.be)
	    - [bair sem feb 2020 livestream](https://www.youtube.com/watch?v=x6AHX-VrcdM&feature=youtu.be)
	- [grad slam 2019](https://www.youtube.com/watch?v=6VVppY-uUgE&feature=youtu.be&t=5600)
	- [textxd 2019](https://berkeley-haas.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=276daa74-7298-40b1-9503-ab17014b5863)
- misc releases
  - [covid coe article](https://engineering.berkeley.edu/news/2020/04/getting-the-right-equipment-to-the-right-people/)
    - [berkeley science rvw blurb](https://berkeleysciencereview.com/article/ppe-to-the-ppeople-2/) 
  - [bids iml piece](https://bids.berkeley.edu/publications/definitions-methods-and-applications-interpretable-machine-learning)
  - https://uvacsnews.wordpress.com/2017/05/04/chandansingh/
    - https://uvacsnews.wordpress.com/2017/05/12/congratulations-to-our-cs-award-winners/
    - https://twitter.com/CS_UVA/status/860155364347715584
  
  