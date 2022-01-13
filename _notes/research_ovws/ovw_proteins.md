---
layout: notes
title: proteins
category: research
---

{:toc}

**some papers involving proteins and ml, especially predicting protein structure from dna/rna**



# TIPs - using viruses for treatment

- videos
  - [Leor Weinberger: Can we create vaccines that mutate and spread? | TED Talk](https://www.ted.com/talks/leor_weinberger_can_we_create_vaccines_that_mutate_and_spread?language=en#t-798841)
  - [Engineering Viruses - Leor Weinberger - Gladstone Institutes 2016 Fall Symposium - YouTube](https://www.youtube.com/watch?v=Dh0RbiAf2jY)
- 2 issues
  - mutation - viruses mutate, our drugs don't
  - transmission - adherence / deployment - hard to give certain the drugs
- solution: use modified versions of viruses as treatment
  - *therapeutic interfering particles* or *TIPs* are engineered deletion mutants designed to piggyback on a virus and deprive the virus of replication material
    - TIP is like a parasite of the virus - it clips off some of the virus DNA (via genetic engineering)
      - it doesn't contain the code for replication, just for getting into the cell
      - since it's shorter, it's made more efficiently - thus it outcompetes the virus
  - how do TIPs spread?
    - mutations happen because the "copy machine" within a cell makes a particular mutation
      - when a new virus comes along, it makes some of the mutated parts
      - TIPs can't replicate, so they take some of the mutated parts made by the new virus
      - then, the TIP gets copied with the same mutation as the virus and this now spreads
- effect
  - viral load will immediately be lower
  - superspreaders can spread treatment to others

- [Identification of a therapeutic interfering particle—A single-dose SARS-CoV-2 antiviral intervention with a high barrier to resistance](https://www.sciencedirect.com/science/article/pii/S0092867421013192?dgcid=coauthor) (chaturvedi...weinberger, 2021)
  - DIP = defective interfering particle = wild-type virus
  - single administration of TIP RNA inhibits SARS-CoV-2 sustainably in continuous cultures
  - in hamsters, both prophylactic and therapeutic intranasal administration of lipid-nanoparticle TIPs durably suppressed SARS-CoV-2 by 100-fold in the lungs, reduced pro-inflammatory cytokine expression, and prevented severe pulmonary edema




# rna structure prediction

## rna basics

- rna does many things
  - certain structures of RNA lend themselves to catalytic activities
  - tRNA/mRNA convert DNA into proteins
  - RNA also serves as the information storage and replication agent in some viruses
- rna components
  - 4 bases: adenine-uracil (instead of dna's thymine), cytosine-guanine
  - ribose sugar in RNA is more flexible than DNA
- RNA World Hypothesis (Walter Gilbert, 1986) - suggests RNA was precursor to modern life
  - later dna stepped in for info storage (more stable) and proteins took over for catalyzing reactions
- rna structure
  - ![rna_structure](../assets/rna_structure.png)
    - primary - sequence in which the bases are aligned - relatively simple, comes from sequencing
    - **secondary** - 2d analysis of hydrogen bonds between rna parts (double-strands, hairpins, loops) - most of stabilizing free energy comes from here (unlike proteins, where tertiary is most important)
      - most work on "RNA folding" predicts secondary structure from primary structure
      - a lot of this comes from hydrogen bonds between pairs (watson-crick edge)
        - other parts of the pairs (e.g. the Hoogsteen- / CH-edge and the sugar edge) can also form bonds
    - tertiary - complete 3d structure (e.g. bends, twists)
  - ![Screen Shot 2021-12-01 at 6.33.41 PM](../assets/Screen%20Shot%202021-12-01%20at%206.33.41%20PM.png)



## code

- [Galaxy RNA workbench](https://github.com/bgruening/galaxy-rna-workbench)
- [viennaRNA](https://github.com/ViennaRNA/ViennaRNA) - incorporating constraints into predictions
- neural nets
  - [spot rna](https://github.com/jaswindersingh2/SPOT-RNA) / [spot rna2](https://github.com/jaswindersingh2/SPOT-RNA2)
    - uses an Ensemble of Two-dimensional Deep Neural Networks and Transfer Learning
  - [mxfold2](https://github.com/keio-bioinformatics/mxfold2/) (sato et al. 2021)

## data

- [Accurate SHAPE-directed RNA structure determination | PNAS](https://www.pnas.org/content/106/1/97) - chemical probing

## algorithms

- [computational bio book ch 10](https://ocw.mit.edu/ans7870/6/6.047/f15/MIT6_047F15_Compiled.pdf) (kellis, 2016)
  - 2 main approaches to rna folding (i.e. predicting rna structure): 
    - (1) thermodynamic stability of the molecule 
    - (2) probabilistic models 
    - note: can use evolutionary (i.e. phylogenetic) data to improve either of these
      - some RNA changes still result in similar structures
      - consistent mutations - something mutates but structure doesn't change (e.g. AU -> G)
      - compensatory mutations - two mutations but structure doesn't change (e.g. AU -> CU -> CG)
      - incorporate similarities to other RNAs along with something like zuker algorithm
  - thermodynamic stability - uses more domain knowledge
    - dynamic programming approach: given energy value for each pair, minimize total energy by pairing up appropriate base pairs
      - **assumption**: no crossings - given a subsequence $[i,j]$, there is either no edge connecting to the ith base (meaning it is unpaired) or there is some edge connecting the ith base to the kth base where $i < k \leq j$ (meaning the ith base is paired to the kth base)
        - this induces a tree-structure to theh problem
      - [nussinov algorithm](https://www.pnas.org/content/77/11/6309.short) (1978) 
        - ignores stacking interactions between neighboring pairs (i.e. assumes there are no pseudo-knots)
      - [zuker algorithm](https://academic.oup.com/nar/article-abstract/9/1/133/1043226) (1981) - includes stacking energies
  - probabilistic approach - uses more statistical likelihood
    - stochastic context-free grammer (SCFG) is like an extension of HMM that incorporates some RNA constraints (e.g. bonds must happen between pairs)
- [Recent advances in RNA folding - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168165617315195) (fallmann et al. 2017)
  - algorithmic advances
    - algorithmic constraints
      - locality - restrict maximal span of base pairs
      - add in bonus energies as hard or soft constraints
    - rna-rna and rna-protein interactions - there are different algorithms for telling how 2 things will interact
    - newer algorithms take into account some aspects of tertiary structure to predict secondary structure (e.g. motifs, nonstandard base pairs, pseudo-knots)
  - representation
    - `...((((((((...)). .)). .((.((...)). )))))).` "dot-parenthesis" notation - opening and closing represent bonded pairs
  - evaluation
    - computing alignments is non-obvious in these representations
    - centroids - structures with a minimum distance to all other structures in the ensemble of possible structures
    - consensus structures - given a good alignment of a collection of related RNA structures, can compute their consensus structure, (i.e., a set of base pairs at corresponding alignment positions)
- [Folding and Finding RNA Secondary Structure](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2982177/) (matthews et al. 2010)
- loosing the assumption of no knots (e.g. crossings)
  - project to topology (e.g. low/high genus) - e.g. projecting to a torus can remove crossings

- [RNA secondary structure prediction using deep learning with thermodynamic integration | Nature Communications](https://www.nature.com/articles/s41467-021-21194-4) (sato et al. 2021)
  - RNA-folding scores learnt using a DNN are integrated together with Turner’s nearest-neighbor free energy parameters
    - DNN predicts scores that are fed into zuker-style dynamic programming

## 3d rna structure prediction

- startup [atomic ai](https://atomic.ai/) works on this
- [Geometric deep learning of RNA structure | science](https://www.science.org/doi/10.1126/science.abe5650) (townshend, ..., dror 2021)
  - scoring model - gives energy function for (primary sequence, structure) pair
- [Frontiers | RNA 3D Structure Prediction Using Coarse-Grained Models | Molecular Biosciences](https://www.frontiersin.org/articles/10.3389/fmolb.2021.720937/full) (li & chen, 2021)
- older
  - [iFoldRNA: three-dimensional RNA structure prediction and folding - PubMed](https://pubmed.ncbi.nlm.nih.gov/18579566/) (sharma et al. 2008)

# protein structure prediction (dna)

## protein basics

- the standard way to obtain the 3D structure of a protein is X-ray crystallography. It takes about a year and costs about $120,000 to obtain the structure of a single protein through X-ray crystallography [[source](https://fortune.com/2020/11/30/deepmind-protein-folding-breakthrough/)]
- on average, a protein is composed of 300 amino acids (residues)
  - 21 amino acid types
  - the first residue is fixed
- proteins evolved via mutations (e.g. additions, substitutions)
  - fitness selects the best one

## algorithm basics

- [multiple-sequence-alignment](https://www.sciencedirect.com/topics/medicine-and-dentistry/multiple-sequence-alignment) (MSA) - alignment of 3 or more amino acid (or nucleic acid) sequences, which show conserved regions within a protein family which are of structural and functional importance
  - matrix is (n_proteins x n_amino_acids)
  - independent model (PSSM) - model likelihood for each column independently
  - pairwise sites (potts model) - model likelihood for pairs of positions
- prediction problems
  - contact prediction - binary map for physical contacts in the final protein
    - common to predict this, evaluation uses binary metrics
  - [Evaluating Protein Transfer Learning with TAPE](https://arxiv.org/abs/1906.08230) - given embeddings, five tasks to measure downstream performance (rao et al. 2019)
    - structure (SS + contact), evolutionary (homology), engineering (fluorescence + stability)
  - future: interactions between molecules (e.g. protein-protein, environment, highly designed proteins)

## deep-learning papers

- [De novo protein design by deep network hallucination | Nature](https://www.nature.com/articles/s41586-021-04184-w) (anishchenko...baker, 2021)
  - deep networks trained to predict native protein structures from their sequences can be inverted to design new proteins, and such networks and methods should contribute alongside traditional physics-based models to the de novo design of proteins with new functions.
- [Highly accurate protein structure prediction with AlphaFold | Nature](https://www.nature.com/articles/s41586-021-03819-2) (jumper, ..., hassabis, 2021)
  - [supp](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf)
  - [best blog post](https://www.blopig.com/blog/2021/07/alphafold-2-is-here-whats-behind-the-structure-prediction-miracle/) ([other blog post](https://towardsdatascience.com/unfolding-alphafold-683d576a54a3))
  - model overview
    - ![alphafold](../assets/alphafold.png)
  - preprocessing
    - MSA
    - finding templates - find similar proteins to model "pairs of residues" - which residues are likely to interact with each other
  - evoformer
    - uses attention on graph network
    - iterative
  - structure model - converts msa/pair representations into set of (x,y,z) coordinates
    - "invariant point attention" - invariance to translations and rotations
- [MSA Transformer](https://proceedings.mlr.press/v139/rao21a.html) (rao et al. 2021) - predicts contact prediction
  - train via masking
  - output (contact prediction) is just linear combination of attention heads finetuned on relatively little data
  - some architecture tricks
    - attention is axial - applied to rows / columns rather than entire input



## covid

- [The Coronavirus Nucleocapsid Is a Multifunctional Protein](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4147684/) (mcbride et al. 2014)
  - coronavirus nucleocapsid = N protein




# bin random papers

- [DeepMind AI tackles one of chemistry’s most valuable techniques](https://www.nature.com/articles/d41586-021-03697-8?utm_source=Nature+Briefing&utm_campaign=b0bda6ba48-briefing-dy-20211210&utm_medium=email&utm_term=0_c9dfd39373-b0bda6ba48-45837550)
- [DeepMind’s AI helps untangle the mathematics of knots](https://www.nature.com/articles/d41586-021-03593-1?utm_source=Nature+Briefing&utm_campaign=b0bda6ba48-briefing-dy-20211210&utm_medium=email&utm_term=0_c9dfd39373-b0bda6ba48-45837550)
