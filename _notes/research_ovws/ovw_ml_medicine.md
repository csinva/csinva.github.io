---
layout: notes
title: ml in medicine
category: research
typora-copy-images-to: ../assets
---

**some rough notes on ml in medicine**

{:toc}


# general

- 3 types
	- disease and patient categorization (e.g. classification)
	- fundamental biological study
	- treatment of patients
- philosophy
  - want to focus on problems doctors can't do
  - alternatively, focus on automating problems parents can do to screen people at home in cost-effective way
- **pathology** - branch of medicine where you take some tissue from a patient (e.g. tumor), look at it under a microscope, and make an assesment of what the disease is
- websites are often easier than apps for patients
- [The clinical artificial intelligence department: a prerequisite for success](https://informatics.bmj.com/content/27/1/e100183) (cosgriff et al. 2020) - we need designated departments for clinical ai so we don't have to rely on 3rd-party vendors and can test for things like distr. shift
- [challenges in ai healthcare (news)](https://www.statnews.com/2019/06/19/what-if-ai-in-health-care-is-next-asbestos/)
  - adversarial examples
  - things can't be de-identified
  - algorithms / data can be biased
  - correlation / causation get confused
- healthcare is 20% of US GDP
- **prognosis** is a guess as to the outcome of **treatment**
- **diagnosis** is actually identifying the problem and giving it a name, such as depression or obsessive-compulsive disorder
- AI is a technology, but it's not a product
- health economics incentives align with health incentives: catching tumor early is cheaper for hospitals

## high-level

- focus on building something you want to deploy
  - clinically useful - more efficient, cutting costs?
  - effective - does it improve the current baseline
  - focused on patient care - what are the unintended consequences
- need to think a lot about regulation
  - USA: FDA
  - Europe: CE (more convoluted)
- intended use
  - very specific and well-defined

## criticisms

- [Dissecting racial bias in an algorithm used to manage the health of populations ](https://science.sciencemag.org/content/366/6464/447)(obermeyer et al. 2019)

# medical system

## evaluation

- doctors are evaluated infrequently (and things like personal traits are often included)
- US has pretty good care but it is expensive per patient
- expensive things (e.g. Da Vinci robot)
- even if ml is not perfect, it may still outperform some doctors

## medical education

- rarely textbooks (often just slides)
- 1-2% miss rate for diagnosis can be seen as acceptable
- [how doctors think](https://www.newyorker.com/magazine/2007/01/29/whats-the-trouble)
  - 2 years: memorizing facts about physiology, pharmacology, and pathology
  - 2 years learning practical applications for this knowledge, such as how to decipher an EKG and how to determine the appropriate dose of insulin for a diabetic
  - little emphasis on metal logic for making a correct diagnosis and avoiding mistakes
  - see work by pat croskerry
  - there is limited data on misdiagnosis rates
  - **representativeness** error - thinking is overly influenced by what is typically true
  - **availability** error - tendency to judge the likelihood of an event by the ease with which relevant examples come to mind
    - common infections tend to occur in epidemics, afflicting large numbers of people in a single community at the same time
    - confirmation bias
  - **affective** error - decisions based on what we wish were true (e.g. caring too much about patient)
  - See one, do one, teach one - teaching axiom

## political elements

- [why doctors should organize](https://www.newyorker.com/culture/annals-of-inquiry/why-doctors-should-organize)
- big pharma
- day-to-day
  - Doctors now face a burnout epidemic: thirty-five per cent of them show signs of high depersonalization
  - according to one recent [report](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2730353?resultClick=1), only thirteen per cent of a physician’s day, on average, is spent on doctor-patient interaction
  - [study](https://www.nytimes.com/2017/11/14/well/live/the-patients-vs-paperwork-problem-for-doctors.html) during an average, eleven-hour workday, six hours are spent at the keyboard, maintaining electronic health records.
  - medicare's r.v.u - changes how doctors are reimbursed, emphasising procedural over cognitive things
  - ai could help - make simple diagnoses faster, reduce paperwork, help patients manage their own diseases like diabetes
  - ai could also make things worse - hospitals are mostly run by business people

# medical communication

## "how do doctors think?"

- easy to misinterpret things to be causal
- often no intuition for even relatively simple engineered features, such as averages
- doctors require context for features (e.g. this feature is larger than the average)
- often have some rules memorized (otherwise memorize what needs to be looked up)
  - unclear how well doctors follow rules
  - some rules are 1-way (e.g. only follow it if it says there is danger, otherwise use your best judgement)
    - 2-way rules are better
    - without proper education 1-way rules can be dangerously used as 2-way rules
    - doesn't make sense to judge 1-way rules on both sepcificity and sensitivity
- rules are often ambiguous (e.g. what constitutes vomiting)
- **doctors adapt to personal experience** - may be unfair to evaluate them on larger dataset
- sometimes said that doctors know 10 medications by heart
- [Overconfidence in Clinical Decision Making](https://www.amjmed.com/article/S0002-9343(08)00152-6/pdf) (croskerry 2008)
  - most uncertainty: family medicine [FM] and emergency medicine [EM]
  - some uncertainty: internal medicine
  - little uncertainty: specialty disciplines
  - 2 systems at work: intuitive (uses context, heuristics) vs analytic (systematic, rule-based)
    - a combination of both performs best
  - doctors are often black boxes as well - validated infrequently, unclear how closely they follow rules
  - doctors adapt to local conditions - should be evaluated only on local dataset
-  [potential liabilities for physicians using ai](https://jamanetwork.com/journals/jama/fullarticle/2752750) (price et al. 2019)
- [What's the trouble. How doctors think](https://www.newyorker.com/magazine/2007/01/29/whats-the-trouble). New Yorker. 2007
- [JAMA Users’ Guide to the Medical Literature](https://jamanetwork.com/journals/jama/article-abstract/192850?utm_campaign=articlePDF&utm_medium=articlePDFlink&utm_source=articlePDF&utm_content=jamapediatrics.2019.1075)
- [TRIPOD 22 points paper](https://www.tripod-statement.org/Portals/0/Tripod Checklist Prediction Model Development and Validation PDF.pdf)
- [basic stats in the step1 exam](https://step1.medbullets.com/topic/dashboard?id=101&specialty=101)
- [How to Read Articles That Use Machine Learning: Users’ Guides to the Medical Literature](https://jamanetwork.com/journals/jama/fullarticle/2754798) (liu et al. 2019
- [Carmelli et al. 2018](https://www.annemergmed.com/article/S0196-0644(18)30327-5/pdf) - primer for CDRs but also a good example of what sort of article I have envisioned creating.
- [Looking through the retrospectoscope: reducing bias in emergency medicine chart review studies.](https://www.ncbi.nlm.nih.gov/pubmed/24746846) (kaji et al. 2018)

## communicating findings

- [don't use ROC curves, use deciles](https://modelplot.github.io/intro_modelplotpy.html)
- [need to evaluate use, not just metric](https://jamanetwork.com/journals/jama/fullarticle/2748179)
- internal/external validity = training/testing error
- model -> fitted model
- retrospective (more confounding, looks back) vs prospective study
- internal/external validity = train/test (although external was usually using different patient population, so is stronger)
- specificity/sensitivity = precision/recall

# examples

## succesful examples of ai in medicine

- [ECG](https://www.nejm.org/doi/full/10.1056/NEJM199112193252503) (NEJM, 1991)
- EKG has a small interpretation on it
- there used to be bayesian networks / expert systems but they went away...

## icu interpretability example

- goal: explain the model not the patient (that is the doctor’s job)
- want to know interactions between features
- some features are difficult to understand
  - e.g. max over this window, might seem high to a doctor unless they think about it
- some features don’t really make sense to change (e.g. was this thing measured)
- doctors like to see trends - patient health changes over time and must include history
- feature importance under intervention

## high-performance ai studies

- chest-xray: chexnet
- echocardiograms: madani, ali, et al. 2018
- skin: esteva, andre, et al. 2017
- pathology: campanella, gabriele, et al.. 2019
- mammogram: kerlikowske, karla, et al. 2018



# medical imaging

- [Medical Imaging and Machine Learning](https://arxiv.org/pdf/2103.01938.pdf)
  - medical images often have multiple channels / are 3d - closer to video than images

# improving medical studies

- Machine learning methods for developing precision treatment rules with observational data (Kessler et al. 2019)
  - goal: find precision treatment rules
  - problem: need large sample sizes but can't obtain them in RCTs
  - recommendations
    - screen important predictors using large observational medical records rather than RCTs
      - important to do matching / weighting to account for bias in treatment assignments
      - alternatively, can look for natural experiment / instrumental variable / discontinuity analysis
      - has many benefits
    - modeling: should use ensemble methods rather than individual models

# pathology

## basics

- **pathologists** work with tissue samples either visually or chemically
   - *anatomic* pathology relies on the microscope whereas *clinical* pathology does not
- pathologists convert from tissue image into written report
- when case is challenging, may require a second opinion (v rare)
- steps (process takes 9-12 hrs): ![tissue_prep](/Users/chandan.singh/Desktop/csinva.github.io/_notes/assets/tissue_prep.png)
   - tissue is surgically removed
      - more tissue collected is generally better (gives more context)
      - this procedure is called a *biopsy*
      - much is written down at this step (e.g. race, gender, locations in organ, different tumors in an organ) that can't be seen in slide alone
   - fixation: keeps the tissue stable (preserves dna also) - basicallly just soak in formalin
   - dissection: remove the relevant part of the tissue
   - tissue processor - removes water in tissue and substitute with wax (parafin) - hardens it and makes it easy to cut into thin strips
   - microtone - cuts very thin slices of the tissue (2-3 microns)
   - staining
      - **H & E** - hematoxylin and eosin stain - most popular (~80%) - colors the cells in a specific way, bc cells are usually pretty transparent
         - hematoxylin stains nucleic acids blue
         - eosin stains proteins / cytoplasm pink/red
      - **immunohistochemistry (IHC)** - tries to identify cell lineage: 10-15%
         - identifies targets
         - use antibodies tagged with chromophores to tag tissues
      - gram stain - highlights bacteria
      - giemsa - microorganisms
      - others...for muscle, fungi
   - viewing
      - usually analog - put slide on something that can move / rotate
      - with paige: put slide through digital scanner (only 5% or so of slides are currently digital)
   - later on, board meets to decide on treatment (based on pathology report)
      - usually some discussion betweeon original imaging (pre-biopsy) and pathologist's interpretation
   - resection - after initial diagnosis, often entire tumor is removed (**resection**)
- how can ai help?
	- can help identify small things in large images
	- can help with conflict resolution
- after (successful) neoadjuvant chemotherapy, problem becomes more difficult
   - very few remaining cancer cells
   - cancer/non-cancer cells become harder to distinguish (esp. for prostate)
   - tumor bed is patchily filled with cancer cells - need to better clarify presence of cancer


## papers

- [Deep Learning Models for Digital Pathology](https://arxiv.org/abs/1910.12329) (BenTaieb & Hamarneh, 2019)
   - note: alternative to histopathology are more expensive / slower (e.g. molecular profiling)
   - to promote consistency and objective inter-observer agreement, most pathologists are trained to follow simple algorithmic decision rules that sufficiently stratify patients into reproducible groups based on tumor type and aggressiveness
   - magnification usually given in microns per pixel
   - WSI files are much larger than other digital images (e.g. for radiology)
   - DNNs can be used for many tasks: beyond just classification, there are subtasks (e.g. count histological primitives, like nuclei) and preprocessing tasks (e.g. stain normalization)
   - challenge: multi-magnification + **high dimensions** (i.e. millions of pixels)
      - people usually extract smaller patches and train on these
         - this loses larger context
         - one soln: pyramid representation: extract patches at different magnification levels
         - one soln: stacked CNN - train fully-conv net, then remove linear layer, freeze, and train another fully-conv net on the activations (so it now has larger receptive field)
         - one soln: use 2D LSTM to aggregate patch reprs.
      - challenge: annotations only at the entire-slide level, but must figure out how to train individual patches
         - e.g. use aggregation techniques on patches - extract patch-wise features then do smth simple, like random forest
         - e.g. treat as weak labels or do multiple-instance learning
            - could just give slide-level label to all patches then vote
      - can use transfer learning from related domains with more labels
   - challenge: class imbalance
      - can use boosting approach to increase the likelihood of sampling patches that were originally incorrectly classified by the model
   - challenge: need to integrate in other info, such as genomics
   - when predicting histological primitives, often predict pixel-wise probability maps, then look for local maxima
      - can also integrated domain-knowledge features
      - can also have 2 paths, one making bounding-box proposals and another predicting the probability of a class
      - alternatively, can formulate as a regression task, where pixelwise prediction tells distance to nearest centroid of object
      - could also just directly predict the count
   - can also predict survival analysis 
- [Clinical-grade computational pathology using weakly supervised deep learning on whole slide images](https://www.nature.com/articles/s41591-019-0508-1) (campanella et al. 2019)
   - use slide-level diagnosis as "weak supervision" for all contained patches
   - 1st step: train patch-level CNNs using MIL
      - if label is 0, then all patches should be 0
      - if label is 1, then only pass gradients to the top-k predicted patches
   - 2nd step: use RNN (or another net) to combine info across *S* most suspicious tiles
- [Human-interpretable image features derived from densely mapped cancer pathology slides predict diverse molecular phenotypes](https://www.nature.com/articles/s41467-021-21896-9) (diao et al. 21)



# cancer

- **tumor** = neoplasm - a mass formation from an uncontrolled growth of cells
   - benign tumor - typically stays confined to the organ where it is present and does not cause functional damage
   - malignant tumor = cancer - comprises organ function and can spread to other organs (**metastasis**)
- relation network based aggregator on patches
- lymphatic system drains fluids (non-blood) from organs into *lymph nodes*
   - cancer often mestastasize through these
- elements of staging pTNM
   - size / depth of tumor "T"
   - number of lymph nodes / how many had cancer "T"
   - number of metastatic foci in non-lymph node organ "M"



## treatments

- chemo
   - traditional chemotherapy disrupts cell replication
      - hair loss and gastrointestinal symptoms occur bc these cells also rapidly replicate
   - *adjuvant* chemotherapy - after cancer is removed, most common
   - *neoadjuvant* chemo - after biopsy, but before resection (when very hard to remove)
- targeted therapies
   - ex. address genetic aberration found in cancer cells
   - immunotherapy - enhance body's immune response to cancer cells (so body will attack these cells on its own)
      - want the antigens on the tumor to be as different as possible (so they will be characterized as foreign)
      - to measure this, can conduct total mutational burden (TMB) or miscrosatellite instability (MSI) test
         - genetic tests - hard to do by looking at glass slide
      - some tumors express receptors (e.g. CTLA4, PD1) that shut off immune cells - some drugs try to block these receptors
