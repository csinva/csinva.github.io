---
layout: notes_without_title
section-type: notes
title: making quick beautiful slides
category: blog
---


# making quick and beautiful slides with reveal-md
**chandan singh**  
*last updated jul 10, 2018*

---

[Reveal-md](https://github.com/webpro/reveal-md) is a library for converting markdown files into slides. If you're familiar with markdown, it can render slides from your markdown files without the need for formatting.

This has 2 main benefits:

1. Quickly making slides, especially with many equations
2. Having a 2D layout that helps keep slides organized
3. You can easily embed webpages / interactive animations

You can see [a demo](https://csinva.github.io/pres/189/#/) of one of my slide decks, created completely from markdown (with external images). 

To make things even faster, I use this `preproc.js` file to automatically  create new slides with a double \## and new columns of slides with a single \#:

```javascript
module.exports = (markdown, options) => {
  return new Promise((resolve, reject) => {
    return resolve(
      markdown
        .split('\n')
        .map((line, index) => {
          if (/### /.test(line))
              return line
          else if (/## /.test(line)) 
              return  '\n--\n' + line;
          else if (/# /.test(line)) 
              return  '\n---\n' + line;
          else
              return line
        })
        .join('\n')
    );
  });
};
```