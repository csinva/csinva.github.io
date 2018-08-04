cd /Users/chandan/website/_notes/ref/ai_slides
reveal-md --preprocessor preproc.js ai_slides.md --static _site
rm -r -f /Users/chandan/website/pres/188
mv _site /Users/chandan/website/pres/188
git add /Users/chandan/website/pres/188
git commit -m 'update ai slides'
git push
exit