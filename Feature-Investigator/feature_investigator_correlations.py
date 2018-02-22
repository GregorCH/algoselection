#!python

from numpy import *
import matplotlib.pyplot as plt
import os, string, sys
import numpy as np
import math
import csv
import numbers
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib.patches import FancyArrowPatch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
from sklearn.metrics import euclidean_distances
from sklearn import manifold

# ex. of how to run code:
# python feature_investigator_correlations.py mipdev-features.csv
# each row in mipdev-features in an instance, the columns are features
# questions can be sent to Alex Georges: ageorges@ucsd.edu

###################################################################

# Section 1: Grabs Feature Data

###################################################################

FILE=sys.argv[1]

with open("%s" %(FILE)) as f:
	reader = csv.reader(f, delimiter=",")
	data = list(reader)

instances_mip = [os.path.basename(line[0]).split('.')[0] for line in data[1:]]
features_mip = [line[1:-1] for line in data[1:]]

## Normalize the Data
#features_scaled = preprocessing.normalize(features_mip) #scale feature vectors to unit norm
#features_scaled = preprocessing.maxabs_scale(features_mip) #scale features to [-1.1]
features_scaled = preprocessing.scale(features_mip) #features centered around zero and scaled to unit variance

###################################################################

# Section 2A: Pearson Correlation Heatmap

###################################################################



corr=np.corrcoef(features_scaled,rowvar=False)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=data[0][1:], yticklabels=data[0][1:])
plt.xticks(rotation=90)
plt.tick_params(labelsize=6)
plt.yticks(rotation=0)
plt.title("Feature Pearson Correlation Heatmap")
plt.savefig('Feature Correlation Heatmap.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


###################################################################
###################################################################
###################################################################
































## Haiku ##
# Flannel and mathy 
# A social one loves to MIP
# David from Davis
# -- "David" by Alex 
