---
layout: blog
title: test
---


From https://github.com/pcooksey/bibtex-js/wiki/Styles


I wanna cite <div style="color=red" class="bibtex_display" bibtexkeys="book1"></div>




<div class="bibtex_template">
  <div class="if author" style="font-weight: bold;">
    <span class="if year">
      <span class="year"></span>, 
    </span>
    <span class="author"></span>
    <span class="if url" style="margin-left: 20px">
      <a class="url" style="color:red; font-size:10px">(view online)</a>
    </span>
  </div>
  <div style="margin-left: 10px; margin-bottom:5px;">
    <span class="title"></span>
  </div>
</div>

<bibtex src="test.bib"></bibtex>

<style>
.year {
    color:red;
}
</style>
