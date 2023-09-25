---
layout: notes
title: great slides with reveal markdown
category: blog
---

**This post and the [associated code](https://github.com/csinva/csinva.github.io/tree/master/_blog/reveal_md_enhanced) contain some enhancements for [reveal-md](https://github.com/webpro/reveal-md), an awesome library for creating presentations with markdown.**

This has several big benefits:

1. Quickly making slides, especially with many equations
2. Having a 2D layout that helps keep slides organized
3. You can easily embed webpages / interactive animations

# demo

You can see [a demo](https://csinva.github.io/pres/189/#/) of one of my reveal-md presentations for teaching machine learning, created completely from markdown (with external images). 


# enhancements

To make things even faster, I use this `preproc.js` file to automatically  create new slides with two hashtags (\##) and new columns of slides with a single hashtag (\#). To use, run with `reveal-md --preprocessor preproc.js slides.md`

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

The `slides.md` file shows a number of non-obvious ways to improve reveal-md. Some examples are:

- inserting blank headers using `## <div> </div>`
- inserting iframes using inline html 
```css
    <div class="divmomentum">
        <iframe class="iframemomentum" src="https://distill.pub/2017/momentum/" scrolling="no" frameborder="no"></iframe>
    </div>

    <style>
    .divmomentum {
        position: relative;
        width: block;
        height: 600px;
        overflow: hidden;
    }

    .iframemomentum {
        position: absolute;            
        top: -165px;
        left: -25px;
        width: 1424px;
        height: 768px;
    }
    </style>
```

- styling the presentation using css inline reveal-md file:

```css
    <style>
    .reveal h1,
    .reveal h2,
    .reveal h3,
    .reveal h4,
    .reveal h5,
    .reveal h6 {
        text-transform: lowercase;
    }
    .reveal section img { 
        background:none; 
        border:none; 
        box-shadow:none; 
        filter: invert(1); 
    }
    iframe {
        filter: invert(1);
    }
    body {
      background: #000;
      background-color: #000; 
    }
    </style>
```