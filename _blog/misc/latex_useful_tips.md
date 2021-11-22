---
layout: notes
title: latex tips and tricks
category: blog
---


*Contains simple instructions for common issues with writing in latex*

## introduction

- really good introduction [here](https://www.overleaf.com/learn/latex/Free_online_introduction_to_LaTeX_(part_1))



## diffing complex latex files

**Pipeline for finding the differences between 2 versions of complex latex projects.**

*Note: this assumes that style files for the two do not change (only changes to .tex files).*

1. Download all files for the original version and the new version into folders

2. "Flatten" the tex files by replacing `\input{fname.tex}` with the contents of the file `fname.tex`
  - simple python code to do this [here](https://github.com/johnjosephhorton/flatex)
  - installing (locally): 
  ```bash
      git clone git@github.com:johnjosephhorton/flatex.git
      cd flatex
      pip install --editable . 
  ```
  - running: ```flatex --include_bbl inputfile.tex outputfile.tex```


3. Copy the non-tex files (.cls, .bib, .sty, .bst, figs) you want to use to a new directory

4. From this directory run [latexdiff](https://github.com/ftilmann/latexdiff/) on the flattened files - this should produce the desired diff!
  - running: `latexdiff file1_flattened.tex file2_flattened.tex > diff.tex`




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
