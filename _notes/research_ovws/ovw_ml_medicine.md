---
layout: notes
title: ml in medicine
category: research
typora-copy-images-to: ../assets
---

**some rough notes on ml in medicine**

{:toc}

# datasets

- [physionet](https://physionet.org/about/database/)
  - mimic-iv
- [nih datasets](https://nda.nih.gov/)
- [mdcalc datasets](https://www.mdcalc.com/)
- [pecarn](https://pecarn.org/datasets/)
- [openneuro](https://openneuro.org/)

## nlp

- [n2c2 tasks](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)
  - MedNLI - NLI task grounded in patient history ([romanov & shivade, 2018](https://arxiv.org/abs/1808.06752))
    - derived from Mimic, but expertly annotated
  - [i2b2](https://www.i2b2.org/NLP/DataSets/) named entity recognition tasks
    - i2b2 2006, 2010, 2012, 2014
- [CASI dataset](https://conservancy.umn.edu/handle/11299/137703) - collection of abbreviations and acronyms (short forms) with their possible senses (long forms), along with other corresponding information about these terms
  - some extra annotations by [agrawal...sontag, 2022](https://arxiv.org/abs/2205.12689)
- [PMC-Patients](https://arxiv.org/abs/2202.13876) - open-source patient snippets, but no groundtruth labels besides age, gender
- [EBM-NLP](https://github.com/bepnye/EBM-NLP) - annotates PICO (Participants, Interventions, Comparisons and Outcomes) spans in clinical trial abstracts
  - task - identify the spans that describe the respective PICO elements.

- mimic-iv-benchmark ([xie...liu, 2022](https://www.nature.com/articles/s41597-022-01782-9))
  - 3 tabular datasets derived from MIMIC-IV ED EHR
    - *hospitalization* (versus discharged) -  met with an inpatient care site admission immediately following an ED visit
    - *critical* - inpatient portality / transfer to an ICU within 12 hours
    - *reattendance* - patient's return visit to ED within 72 hours

  - preprocessing for outliers / missing values (extended descriptions of variables [here](https://static-content.springer.com/esm/art%3A10.1038%2Fs41597-022-01782-9/MediaObjects/41597_2022_1782_MOESM1_ESM.pdf))
    - patient history
      - past ed visits, hospitalizations, icu admissions, comorbidities
      - ICD codes give patients comorbidities (CCI charlson comorbitidy index, ECI elixhauser comorbidity index)

    - info at triage
      - temp., heart rate, pain scale, ESI, ...
      - Emergency severity index (ESI) - 5-level triage system assigned by nurse based on clinical judgments (1 is highest priority)
      - top 10 chief complaints
      - No neurological features (e.g. GCS)

    - info before discharge
      - vitalsigns
      - edstays
      - medication prescription

## ucsf de-id data

- black-box
  - predict [postoperative delirium prediction](https://bmcanesthesiol.biomedcentral.com/articles/10.1186/s12871-021-01543-y) (bishara, ..., donovan, 2022)
- intrepretable
  - predict multiple sceloris by incorporating domain knowledge into biomedical knowledge graph ([nelson, ..., baranzini, 2022](https://academic.oup.com/jamia/article/29/3/424/6463510))
  - predict mayo endoscopic subscores from colonoscopy reports ([silverman, ..., 2022](https://www.medrxiv.org/content/10.1101/2022.06.19.22276606.abstract))

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
- The impact of inconsistent human annotations on AI driven clinical decision making ([sylolypavan...sim, 2023](https://www.nature.com/articles/s41746-023-00773-3)) - labels / majority vote are often very inconsistent

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