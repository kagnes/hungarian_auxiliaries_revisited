#!/usr/bin/env python
# coding: utf-8


# created: 2023/03/18
# published: 2023/09/23
# based on this tutorial: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/


# import modules
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from adjustText import adjust_text


# read dataset
df = pd.read_csv('input_similarity_matrix.tsv', sep='\t', index_col=0)

# drop the frequency column (it is not a separate feature)
df = df.drop(['freq'], axis=1)

# print the shape of the dataset (rows = finite forms, columns = features)
print(df.shape)

# print a few data points
print(df.head(4))

# normalize the feature values
X = normalize(df.values+1.0)

# which set of parameters would be the best for the hierarchical agglomerative clustering?
# try every method-metrics combination available in SciPy
# calculate the cophenetic correlation coefficient (cophenet) for each resulting matrix
# choose the parameters producing the largest cophenet value

methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
metrics = ['euclidean', 'minkowski', 'cityblock', 'seuclidean', 'sqeuclidean', 'cosine', 'correlation', 'hamming', 
           'jaccard', 'jensenshannon', 'chebyshev', 'canberra', 'braycurtis', 'mahalanobis', 'yule', 'matching', 
           'dice', 'kulczynski1', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath']

combinations = []
for i in range(len(methods)):
    for j in range(len(metrics)):
        combinations.append((methods[i], metrics[j]))

candidates = []
for combo in combinations:
    try:
        Z = linkage(X, method=combo[0], metric=combo[1])
        c, coph_dists = cophenet(Z, pdist(X))
        if not np.isnan(c):
            candidates.append([combo[0], combo[1], c.item()])
    except ValueError:
        pass

candidates.sort(key=lambda x: x[2], reverse=True)
hc = candidates[0]
print('Method-metric combination with the highest score: {}, {}, {}'.format(hc[0], hc[1], hc[2]))

# create the final linkage matrix using the best method-metric combination
Z = linkage(X, method=hc[0], metric=hc[1])

# generate the full dendrogram
plt.figure(figsize=(5, 20))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('distance')
plt.ylabel('finite form')
dendrogram(Z, labels=df.index, leaf_font_size=8., color_threshold=0.7, orientation='right', count_sort='descending')
plt.show()

# set the cutoff manually, based on the figure
cutoff = 0.7

# add colors to clusters
colors = fcluster(Z, cutoff, criterion='distance')
categories = defaultdict(list)
for i, c in enumerate(colors):
    categories[c].append(df.index[i])
for k in sorted(categories):
    print(k, ' '.join(sorted(categories[k])))

# apply t-SNE and create a 2D-representation of the clusters
emb = TSNE().fit_transform(X)

m = ['o', '*', '^', 'x']
zippedx = zip(*[emb[:,0], colors])
zippedy = zip(*[emb[:,1], colors])

groupdictx = {}
for z in zippedx:
    if z[-1] not in groupdictx.keys():
        groupdictx[z[-1]] = []
    groupdictx[z[-1]].append(z)
    
groupdicty = {}
for z in zippedy:
    if z[-1] not in groupdicty.keys():
        groupdicty[z[-1]] = []
    groupdicty[z[-1]].append(z)

groupsx = []
groupsy = []
for k, v in groupdictx.items():
    groupsx.append([i[0] for i in v])
for k, v in groupdicty.items():
    groupsy.append([i[0] for i in v])

scatters = []
plt.figure(figsize=(8, 5))
for i in range(len(groupsx)):
    sc = plt.scatter(groupsx[i], groupsy[i], marker=m[i], s=20.0)
    scatters.append(sc)
texts = [plt.text(x, y, t) for x, y, t in zip(*[emb[:,0], emb[:,1], df.index])]
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=.5))

plt.legend((scatters[0], scatters[2], scatters[1], scatters[3]), ('1', '2', '3', '4'), scatterpoints=1, loc='upper left', ncol=1, fontsize=10)

plt.xticks([], [])
plt.yticks([], [])
plt.tight_layout()
plt.show()

