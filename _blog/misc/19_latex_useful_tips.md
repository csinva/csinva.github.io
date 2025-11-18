---
layout: notes
title: latex tips and tricks
category: blog
---


*Contains simple instructions for common issues with writing in latex*

## introduction

- really good introduction [here](https://www.overleaf.com/learn/latex/Free_online_introduction_to_LaTeX_(part_1))

## diffing latex projects (easy)

- from overleaf, create a `latexmkrc` file with this content: `$pdflatex = "latexdiff main_old.tex main.tex > main-d.tex; pdflatex %O  main-d"`
  - Save the original version into “main_old.tex” and the current version as “main.tex”
  - This will render a marked up pdf with the differences

## diffing complex latex files (hard)

**Pipeline for finding the differences between 2 versions of complex latex projects.**

*Note: this assumes that style files for the two do not change (only changes to .tex files).*

1. Download all files for the original version and the new version into folders named `old` and `new`

2. "Flatten" the tex files by replacing `\input{fname.tex}` with the contents of the file `fname.tex`
  - simple python code to do this [here](https://github.com/johnjosephhorton/flatex)
  - installing (locally): 
  ```bash
  git clone https://github.com/johnjosephhorton/flatex.git
  cd flatex
  pip install --editable . 
  ```
   ```bash
   cd old
   flatex --include_bbl _main.tex old.tex
   cd ..
   cd new
   flatex --include_bbl _main.tex new.tex
   cd ..
   ```
  
3. Copy the non-tex files (.cls, .bib, .sty, .bst, figs) you want to use to a new directory. Then From this directory download and run [latexdiff](https://github.com/ftilmann/latexdiff/) on the flattened files - this should produce the desired diff!
   ```bash
	 mkdir diff
	 cp -r new/* diff/
	 cp old/old.tex diff/
    
   cd diff
   latexdiff old.tex new.tex > diff.tex
   ```


## toggling documents with a simple boolean

**a simple way to use booleans in latex without needing to import any packages**

defining a boolean: 
```
\newif\ifanonymize % define a boolean named `anonymize`

\anonymizefalse % set it to false
% \anonymizetrue % set it to true
```

use it in an if-statement
```latex
\ifanonymize
	...
\else
	...
\fi
```

create a helper command named `\anonfinal` which takes two arguments and will print only one of them based on the boolean
```latex
\ifanonymize
  \newcommand{\anonfinal}[2]{#1}
\else
  \newcommand{\anonfinal}[2]{#2}
\fi
```



## making figures

- figures should be saved as pdf unless they have a ton of points

## some useful links
- https://github.com/Wookai/paper-tips-and-tricks
