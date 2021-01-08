'''Copies files from notes directory, does some preprocessing, then builds them into a web page using jupyter-book
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess
from os.path import join as oj
from sklearn.feature_extraction.text import TfidfVectorizer

# source / dest
src = '/Users/chandan/website/_notes/'
dest = 'notes'
dest_extraneous = '_build/html/notes'

# copy stuff
try:
    shutil.rmtree(dest)  # rm
except:
    pass
destination = shutil.copytree(src, dest)  # copy
shutil.rmtree(f'{dest}/talks/')  # rm
shutil.rmtree(f'{dest}/misc/')  # rm

# toc
toc = \
    '''- file: intro
      numbered: true
    
    '''

# process the files
contents = []
fnames = []
for folder in os.listdir(dest):
    if not '.' in folder and not 'cheat' in folder and not 'assets' in folder and not 'misc' in folder:
        header_file = folder.lower() + '.md'
        header_fname = oj(dest, folder, header_file)
        open(header_fname, "w").write('# ' + folder)
        toc += f'- file: {header_fname}\n  sections:\n'
        for fname in os.listdir(oj(dest, folder)):
            if fname.endswith('.md') and not fname == header_file:
                fpath = oj(dest, folder, fname)
                content = open(fpath, "r").read().replace('# ', '## ')
                try:
                    title = content.split('title:', 1)[1].split('\n')[0].lower()
                    title = content.split('title:', 1)[1].split('\n')[0].lower()
                    content = content.replace('{:toc}', f'# {title}')
                    content = content.replace('category: ', 'cat: ')  # remove category information
                    open(fpath, "w").write(content)
                    toc += f'  - file: {fpath}\n'
                except:
                    os.remove(fpath)
                contents.append(content)
                fnames.append(fname[:-3])
open('_toc.yml', 'w').write(toc)

# make visualization
fnames = [x.replace('_', ' ') for x in fnames]
vect = TfidfVectorizer(min_df=1, stop_words="english")
tfidf = vect.fit_transform(contents)
pairwise_similarity = (tfidf * tfidf.T).todense()
plt.figure(figsize=(11, 9), dpi=150)
plt.imshow(pairwise_similarity)
plt.yticks(np.arange(71), labels=fnames, fontsize=8)
plt.xticks(np.arange(71), labels=fnames, fontsize=8, rotation='vertical')
plt.ylim((-0.5, 70))
plt.colorbar(label='Similarity (tf-idf)')
plt.savefig('area_similarities.svg')

# jb build .
subprocess.run(['jb', 'build', '.'])

# rm notes
shutil.rmtree(dest)  # rm
shutil.rmtree(dest_extraneous)  # rm
