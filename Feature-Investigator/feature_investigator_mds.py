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
from math import ceil
from pylab import figure

# ex. of how to run code:
# python feature_investigator_mds.py mipdev-features.csv
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

# Section 2A: MDS, Find the number of dimensions to map to
# This can take a few minutes, so feel free to grab a coffee at this point

###################################################################
# Below, "information" is defined as (1-stress/scale)
# The scale here is stress[dimension(2)]
# Information is a scalar in [0,1]
## Stress is defined as the sum of squared difference between 
## distances in the embedded space and distances in the original space

max_dim=ceil(0.25*len(features_mip[0]))

print()
print("Max dimension projecting to is %s" %(max_dim))
print()

stress, dimension=[],[]
fig, ax = plt.subplots()
for i in range(2,max_dim): # choose the range of dimensions to map to
	print('Projecting to dimension %s' %i)
	mds = manifold.MDS(i) # number of dimensions to map to
	proj = mds.fit_transform(features_scaled).T
	stress.append(mds.stress_)
	dimension.append(i)
information = [1-i/stress[0] for i in stress]

print()

n=1
for line in information:
	n+=1
	if line >= 0.7: #set a 70% threshold for information kept
		print('%.3f information is kept at dimension %s' %(line,n))
		break
n=1
for line in information:
	n+=1
	if line >= 0.9: #set a 90% threshold for information kept
		print('%.3f information is kept at dimension %s' %(line,n))
		break	
n=1
for line in information:
	n+=1
	if line >= 0.95: #set a 95% threshold for information kept
		print('%.3f information is kept at dimension %s' %(line,n))
		break			
extraticks=[0.7,0.9,0.95]		
plt.axhline(y=0.7, color='r', linestyle='-')
plt.axhline(y=0.9, color='r', linestyle='-')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.plot(dimension,information,'bo')
plt.yticks(list(plt.yticks()[0])+extraticks)
plt.ylim((-0.1,1.1))
plt.xlabel('Dimension')
plt.ylabel('Information Retained')
plt.title("MDS Normalized Retained Information")
plt.savefig('MDS_information.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

###################################################################

# Section 2B: MDS, Draw 2D and 3D MDS plots

###################################################################
# Draw 2D MDS 
mds = manifold.MDS(2) # number of dimensions to map to
proj = mds.fit_transform(features_scaled).T
fig, ax = plt.subplots()
plt.plot(proj[0],proj[1], 'ro')
plt.title("Feature MDS")
plt.xscale("symlog")
plt.yscale("symlog")
plt.axis('tight')
plt.savefig('Feature MDS.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

# Draw 3D MDS.  3D Images will not save automatically.
mds = manifold.MDS(3) # number of dimensions to map to
proj = mds.fit_transform(features_scaled).T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(proj[0],proj[1], proj[2])
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.set_zscale("symlog")
plt.show()
plt.close()



# Do the same as the above, just with names of instances attached to points
mds = manifold.MDS(2) # number of dimensions to map to
proj = mds.fit_transform(features_scaled).T
fig, ax = plt.subplots()
for i, txt in enumerate(instances_mip):
   ax.annotate(txt, (proj[0][i],proj[1][i]))
plt.plot(proj[0],proj[1], 'ro')
plt.title("Feature MDS")
plt.xscale("symlog")
plt.yscale("symlog")
plt.axis('tight')
plt.savefig('Feature MDS (names).pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()



# The following code is broken.  It's supposed to label 3D MDS
'''
mds = manifold.MDS(3) # number of dimensions to map to
proj = mds.fit_transform(features_scaled).T
fig = figure()
ax = Axes3D
for i, txt in enumerate(instances_mip):
	label = '%s' %(str(i))
	ax.text(proj[0][i],proj[1][i],proj[2][i], label, size=20, zorder=1)
ax.scatter(proj[0],proj[1], proj[2])
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.set_zscale("symlog")
plt.show()
plt.close()
'''



###################################################################
###################################################################
###################################################################



























## Haiku ##
# A voice from heaven
# Eats just the Prague Pierogies
# Not a cartoon, Bart
# -- "Bart" by Alex