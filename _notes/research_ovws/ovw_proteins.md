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
  - viruses mutate, our drugs don't
  - adherence / deployment - hard to give certain the drugs
- solution: use modified versions of viruses as treatment
  - *therapeutic interfering particles* or *TIPs* are engineered deletion mutants designed to piggyback on a virus and deprive the virus of replication material
    - TIP is like a parasite of the virus - it clips off some of the virus DNA (via genetic engineering)
      - it doesn't contain the code for replication, just for getting into the cell
      - since it's shorter, it's made more efficiently - thus it outcompetes the virus
  - how do TIPs spread?
    - mutations happen because the "copy machine" within a cell makes a particular mutation
      - when a new virus comes along, it makes some of the mutate parts
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
      - [nussinov algorithm](https://www.pnas.org/content/77/11/6309.short) (1978) - given a subsequence $[i,j]$, there is either no edge connecting to the ith base (meaning it is unpaired) or there is some edge connecting the ith base to the kth base where $i < k \leq j$ (meaning the ith base is paired to the kth base)
        - ignores stacking interactions between neighboring pairs (i.e. assumeres there are no pseudo-knots)
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
- [RNA secondary structure prediction using deep learning with thermodynamic integration | Nature Communications](https://www.nature.com/articles/s41467-021-21194-4) (sato et al. 2021)
  - RNA-folding scores learnt using a DNN are integrated together with Turner’s nearest-neighbor free energy parameters
    - DNN predicts scores that are fed into zuker-style dynamic programming

# protein structure prediction (dna)

- [De novo protein design by deep network hallucination | Nature](https://www.nature.com/articles/s41586-021-04184-w) (anishchenko...baker, 2021)
  - deep networks trained to predict native protein structures from their sequences can be inverted to design new proteins, and such networks and methods should contribute alongside traditional physics-based models to the de novo design of proteins with new functions.

- [Highly accurate protein structure prediction with AlphaFold | Nature](https://www.nature.com/articles/s41586-021-03819-2) (jumper, ..., hassabis, 2021)
  - [supp](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf)
  - [best blog post](https://www.blopig.com/blog/2021/07/alphafold-2-is-here-whats-behind-the-structure-prediction-miracle/) ([other blog post](https://towardsdatascience.com/unfolding-alphafold-683d576a54a3))
  - background
    - the standard way to obtain the 3D structure of a protein is X-ray crystallography. It takes about a year and costs about $120,000 to obtain the structure of a single protein through X-ray crystallography [[source](https://fortune.com/2020/11/30/deepmind-protein-folding-breakthrough/)]
    - on average, a protein is composed of 300 amino acids (residues)
      - 21 amino acid types
      - the first residue is fixed
  - model overview
    - ![alphafold](../assets/alphafold.png)
  - preprocessing
    - [multiple-sequence-alignment](https://www.sciencedirect.com/topics/medicine-and-dentistry/multiple-sequence-alignment) (MSA) - alignment of 3 or more amino acid (or nucleic acid) sequences, which show conserved regions within a protein family which are of structural and functional importance.
    - finding templates - find similar proteins to model "pairs of residues" - which residues are likely to interact with each other
  - evoformer
    - uses attention on graph network
    - iterative
  - structure model - converts msa/pair representations into set of (x,y,z) coordinates
    - "invariant point attention" - invariance to translations and rotations
