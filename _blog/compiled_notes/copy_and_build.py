'''Copies files from notes directory, does some preprocessing, then builds them into a web page using jupyter-book
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess
from os.path import join as oj
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import seaborn as sns
from tqdm import tqdm

# source / dest
src = '/Users/chandan/website/_notes/'
dest = 'notes'
dests_extraneous = ['_build/html/_sources/', '_build/.doctrees', '_build/html/.buildinfo',
                    '_build/html/notes/readme.html']

# copy stuff
try:
    shutil.rmtree(dest)  # rm
except:
    pass
destination = shutil.copytree(src, dest)  # copy
shutil.rmtree(f'{dest}/talks/')  # rm
shutil.rmtree(f'{dest}/misc/')  # rm

# toc
toc = '''- file: intro\n  numbered: true\n'''

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
                    title = content.split('title:', 1)[1].split('\n')[0].capitalize()
                    content = content.replace('{:toc}', f'# {title}')
                    content = content.replace('category: ', 'cat: ')  # remove category information
                    open(fpath, "w").write(content)
                    toc += f'  - file: {fpath}\n'
                except:
                    os.remove(fpath)
                contents.append(content)
                fnames.append(fname[:-3])
open('_toc.yml', 'w').write(toc)
os.system(
    'jupyter-book toc migrate /Users/chandan/website/_blog/compiled_notes/_toc.yml -o /Users/chandan/website/_blog/compiled_notes/_toc.yml')

# make visualization
fnames = [x.replace('_', ' ').replace('ovw ', '*') for x in fnames]
vect = TfidfVectorizer(min_df=1, stop_words="english")
tfidf = vect.fit_transform(contents)
pairwise_similarity = (tfidf * tfidf.T).todense()
plt.figure(figsize=(11, 9), dpi=150)
plt.imshow(pairwise_similarity)
plt.yticks(np.arange(len(fnames)), labels=fnames, fontsize=8)
plt.xticks(np.arange(len(fnames)), labels=fnames, fontsize=8, rotation='vertical')
plt.ylim((-0.5, 70))
plt.colorbar(label='Similarity (tf-idf)')
plt.savefig('area_similarities.svg')

# make graph visualization
pairwise_similarity = (pairwise_similarity - pairwise_similarity.min()) / (
        pairwise_similarity.max() - pairwise_similarity.min())
plt.figure(figsize=(14, 10))
G = nx.Graph()
for i in range(len(fnames)):
    for j in range(i):
        G.add_edge(fnames[i], fnames[j], weight=np.square(pairwise_similarity[i, j] * 3))

pos = nx.spring_layout(G, seed=7, k=10 / np.sqrt(len(fnames)))  # positions for all nodes - seed for reproducibility
nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.25)
# nx.draw_networkx_edges(G, pos, edgelist=G.edges, alpha=0.1)
print('making graph viz...')
for (u, v, d) in tqdm(G.edges(data=True)):
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1.7 * d['weight'], alpha=0.35)

nx.draw_networkx_labels(G, pos, font_size=13, font_family="sans-serif", font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# nx.draw(G, with_labels=True)
plt.tight_layout()
plt.savefig('similarities_graph.svg')

# jb build .
subprocess.run(['jb', 'build', '.'])
# jb build . --builder pdfhtml # make pdf

# rm notes
try:
    shutil.rmtree(dest)  # rm
    for dest_extraneous in dests_extraneous:
        shutil.rmtree(dest_extraneous)  # rm
except:
    pass

"""

'''Copies files from notes directory, does some preprocessing, then builds them into a web page using jupyter-book
'''

import os
import shutil
import subprocess
from os.path import join as oj

import matplotlib.pyplot as plt

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import seaborn as sns
from tqdm import tqdm

# source / dest
src = '/Users/chandan/website/_notes/'
dest = 'notes'
dests_extraneous = ['_build/html/_sources/', '_build/.doctrees', '_build/html/.buildinfo',
                    '_build/html/notes/readme.html']

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
                    title = content.split('title:', 1)[1].split('\n')[0].capitalize()
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
for i in range(len(fnames)):
    fnames[i] = fnames[i].replace('ovw ', '*')
vect = TfidfVectorizer(min_df=1, stop_words="english")
tfidf = vect.fit_transform(contents)
pairwise_similarity = (tfidf * tfidf.T).todense()
pairwise_similarity = (pairwise_similarity - pairwise_similarity.min()) / (
            pairwise_similarity.max() - pairwise_similarity.min())

plt.figure(figsize=(11, 9), dpi=150)
plt.imshow(pairwise_similarity)
plt.yticks(np.arange(len(fnames)), labels=fnames, fontsize=8)
plt.xticks(np.arange(len(fnames)), labels=fnames, fontsize=8, rotation='vertical')
plt.ylim((-0.5, 70))
plt.colorbar(label='Similarity (tf-idf)')
plt.savefig('area_similarities.svg')
# import seaborn as sns
# cg = sns.clustermap(pairwise_similarity, xticklabels=fnames, yticklabels=fnames,
#                     dendrogram_ratio=0.01, colors_ratio=0.01, cbar_pos=None)
# cg.ax_row_dendrogram.set_visible(False)
# cg.ax_col_dendrogram.set_visible(False)
# plt.savefig('area_similarities.png')


# make mindmap
plt.figure(figsize=(14, 10))
G = nx.Graph()
for i in range(len(fnames)):
    for j in range(i):
        G.add_edge(fnames[i], fnames[j], weight=np.square(pairwise_similarity[i, j] * 3))

pos = nx.spring_layout(G, seed=7, k=10 / np.sqrt(len(fnames)))  # positions for all nodes - seed for reproducibility
nx.draw_networkx_nodes(G, pos, node_size=700)
# nx.draw_networkx_edges(G, pos, edgelist=G.edges, alpha=0.1)
for (u, v, d) in tqdm(G.edges(data=True)):
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1.7 * d['weight'], alpha=0.35)

nx.draw_networkx_labels(G, pos, font_size=13, font_family="sans-serif", font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')

# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# nx.draw(G, with_labels=True)
plt.tight_layout()
# plt.savefig('similarities_graph.png')
plt.savefig('similarities_graph.svg')


# jb build .
subprocess.run(['jb', 'build', '.'])
# jb build . --builder pdfhtml # make pdf

# rm notes
shutil.rmtree(dest)  # rm
for dest_extraneous in dests_extraneous:
    shutil.rmtree(dest_extraneous)  # rm

"""
