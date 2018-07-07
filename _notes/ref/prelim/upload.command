cd /Users/chandan/drive/website/_notes/ref/prelim
reveal-md --preprocessor preproc.js ai_slides.md --static _site
rm -r -f /Users/chandan/drive/website/pres/188
mv _site /Users/chandan/drive/website/pres/188
git add /Users/chandan/drive/website/pres/188
git commit -m 'update ai slides'
git push
exit