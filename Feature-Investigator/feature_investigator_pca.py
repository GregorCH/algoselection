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

# ex. of how to run code:
# python feature_investigator_pca.py mipdev-features.csv
# each row in mipdev-features in an instance, the columns are features
# questions can be sent to Alex Georges: ageorges@ucsd.edu

###################################################################

#Section 1: Grabs Feature Data

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

#Section 2A: PCA, Find the number of dimensions to map to 

###################################################################
# Below, "information" is defined as (sum of variance of components)/(total variance)
# Information is a scalar in [0,1]

n_components=ceil(0.5*len(features_mip[0]))
dim_70=0
dim_90=0
dim_95=0

pca = PCA(n_components)
pca.fit(features_scaled)
proj = pca.transform(features_scaled)



information, dimension=[],[]
fig, ax = plt.subplots()
for i in range(n_components):
	information.append(sum(pca.explained_variance_ratio_[:i+1]))
	dimension.append(i+1)
n=1
for line in information:
	n+=1
	if line >= 0.7: #set a 70% threshold for information kept
		print('%.3f information is kept at dimension %s' %(line,n))
		dim_70=n
		break
n=1
for line in information:
	n+=1
	if line >= 0.9: #set a 90% threshold for information kept
		print('%.3f information is kept at dimension %s' %(line,n))
		dim_90=n
		break	
n=1
for line in information:
	n+=1
	if line >= 0.95: #set a 95% threshold for information kept
		print('%.3f information is kept at dimension %s. %s components '
				'will be saved to text unless otherwise specified' %(line,n, n))
		dim_95=n
		break		
extraticks=[0.7,0.9,0.95]			
plt.axhline(y=0.7, color='r', linestyle='-')
plt.axhline(y=0.9, color='r', linestyle='-')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.plot(dimension,information,'bo')
plt.yticks(list(plt.yticks()[0])+extraticks)
plt.ylim((0,1.1))
plt.xlabel('Dimension')
plt.ylabel('Information Retained')
plt.title("PCA Retained Information")
plt.savefig('PCA_information.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

###################################################################

#Section 2B: PCA, Plot PCAs

###################################################################

# Plot 1D PCAs for the top 3 components
for k in range(3):
	fig, ax = plt.subplots()
	plt.plot(proj[:,k], 'ro')
	plt.title('Features PCA Component %s' %(k+1))
	plt.xlabel("Instances")
	plt.ylabel("Component %s Value" %(k+1))
	plt.yscale("symlog")
	plt.savefig('Feature PCA C%s.pdf' %(k+1), bbox_inches='tight', pad_inches=0)
	plt.show()
	plt.close()

# Plot 2D PCA comparing Component 1 to Component 2
fig, ax = plt.subplots()
plt.plot(proj[:,0],proj[:,1], 'ro')
plt.title('Feature PCA Component 1 vs Component 2')
plt.xlabel("Component 1 Value")
plt.ylabel("Component 2 Value")
plt.xscale("symlog")
plt.yscale("symlog")
plt.savefig('Feature PCA C1 & C2.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

# Do the same as the above, just with names of instances attached to points
for k in range(3):
	fig, ax = plt.subplots()
	for i, txt in enumerate(instances_mip):
	    ax.annotate(txt, (i,proj[:,k][i]))
	plt.plot(proj[:,k], 'ro')
	plt.title('Features PCA Component %s' %(k+1))
	plt.xlabel("Instances")
	plt.ylabel("Component %s Value" %(k+1))
	plt.yscale("symlog")
	plt.savefig('Feature PCA C%s (names).pdf' %(k+1), bbox_inches='tight', pad_inches=0)
	plt.show()
	plt.close()

fig, ax = plt.subplots()
for i, txt in enumerate(instances_mip):
    ax.annotate(txt, (proj[:,0][i],proj[:,1][i]))
plt.plot(proj[:,0],proj[:,1], 'ro')
plt.title('Feature PCA Component 1 vs Component 2')
plt.xlabel("Component 1 Value")
plt.ylabel("Component 2 Value")
plt.xscale("symlog")
plt.yscale("symlog")
plt.savefig('Feature PCA C1 & C2 (names).pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()


###################################################################

#Section 2C: PCA, Stats & save info 

###################################################################


# Save the PCA components to text file
# Num components is set automatically to keep 95% of information
# Change this if needed.
DATA=[]
header = ['PCA %s' %(i+1) for i in range(dim_95)]
DATA.append(header)
components = proj[:,:dim_95]
for line in components:
	DATA.append(line)
np.savetxt('PCA_components.txt', DATA, fmt='%s')


# Grab highest weighted variables in PCA
# Here, the features are grabbed that have the highest total abs(weight) over all components
# Num features is arbitrarily set to grab dim_95 features
pca_eigenvectors = [abs(pca.components_[i]) for i in range(n_components)]
total_weights=[sum(x) for x in zip(*pca_eigenvectors)]
total_weights=np.array(total_weights)
indices = (-total_weights).argsort()[:dim_95]


top_features =[str(data[0][index+1]) for index in indices]
np.savetxt('PCA_top_features.txt', top_features, fmt='%s')

# Here are some statistics about the PCA
# Print this info or save to file if relevant
pca.explained_variance_ratio_
pca.explained_variance_
pca.mean_
pca.n_components_
pca.noise_variance_


###################################################################
###################################################################
###################################################################
































## Haiku ##
# C++ experts
# Gorana and Radovan
# Love milk for lunches
# -- "Gorana" by Alex
