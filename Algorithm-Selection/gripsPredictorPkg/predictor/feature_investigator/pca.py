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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

from .. import config, logger

log = logger._Logger.get_logger(__name__) # set module name for logging
cfg = config.global_config

def investigate(
	FILE,
	scale_features = 'maxabs',
	show = False,
	images_dir = cfg.finvestig_images_dir,
	data_dir = cfg.finvestig_data_dir,
	results_dir = cfg.finvestig_results_dir):

	'''This function produces multiple results, all pertaining to Principal Component Analysis.
	The goals of this function are to:

	  1) Understand the dimensionality of the feature space according to PCA
	  2) Visualize the various components from PCA
	  3) Determine which features are most important according to PCA
	  4) Construct PCA spaces which can be used as new feature files

	There are various outputs of this function which attain these goals.

	:param FILE: Features file.  This should be in CSV format, with column 0 being
	       the instance name and row zero being the names of the features.
	:param scale_features: There are various ways to scale the features data.
	             The scaling is done column-wise (i.e. on each feature individually).
	             default='maxabs'.

	             - maxabs = Scale to [-1,1]
	             - scale = Zero mean and unit stdev
	             - minmax = Translate and scale to [0,1]
	             - normalize = Normalize each feature to unit norm
	             - robust = Shift outliers in according to interquartile range

	:param show: Whether to show images as they are being produced.  default=False
	:param images_dir: Directory to dump images.
	:param data_dir: Directory to dump data.
	:param results_dir: Directory to dump results.
	:return: Various plots, list of top features predicted by PCA, and PCA spaces.
	     The filename of every saved output automatically has the input file
	     names used to produce it. Plots all returned in PDF format and are
	     automatically saved. Text files are automatically saved:

	     - Text 1: Full PCA space. Has the same number of dimensions as the
	       input feature space. Automatically saved to CSV format in data_dir.
	     - Text 2: Reduced PCA space. The size of this is set to the dimension
	       which keeps 95% of the "information". Information here is defined as
	       the percent variance of data explained by the lower dimensional space.
	       Automatically saved to CSV format in data_dir.
	     - Text 3: Reduced by PCA space. This is a subset of the original
	       Feature space, with most important features chosen as the subset.
	       This has the same dimension as the Reduced PCA space. Automatically
	       saved to CSV format in data_dir.
	     - Text 4: List of most important features. The number of most important
	       features is the same as the size of the Reduced PCA space. These
	       features are selected by determining which features have the overall
	       highest weight in PCA. Automatically saved to TXT format in results_dir.

	     - Plot 1: Information retained with respect to dimension being mapped to.
	       x-axis is scaled by heuristically. Change this if plot isn't nice.
	     - Plot 2: PCA in Component 1
	     - Plot 3: PCA in Component 2
	     - Plot 4: PCA in Component 3
	     - Plot 5: PCA in Compoment 1 and Component 2
	     - Plot 6: PCA in Component 1, with names attached to points
	     - Plot 7: PCA in Component 2, with names attached to points
	     - Plot 8: PCA in Component 3, with names attached to points
	     - Plot 9: PCA in Compoment 1 and Component 2, with names attached to points

	'''

	###################################################################
	#Section 1: Grabs Feature Data
	###################################################################

	stamp = '%s' %(os.path.basename(FILE).split('.')[0])

	with open("%s" %(FILE)) as f:
		reader = csv.reader(f, delimiter=",")
		data = list(reader)

	instances_mip = [os.path.basename(line[0]).split('.')[0] for line in data[1:]]
	features_mip = [line[1:] for line in data[1:]]

	###################################################################
	# Section 1B: Scale the feature/performance data
	###################################################################
	# normalize = scale to unit norm
	# maxabs_scale = scale to [-1,1]
	# scale = zero mean scaled to std one

	if scale_features == 'scale':
		features_mip = preprocessing.scale(features_mip)
	elif scale_features == 'maxabs':
		features_mip = preprocessing.maxabs_scale(features_mip)
	elif scale_features == 'minmax':
		features_mip = preprocessing.minmax_scale(features_mip)
	elif scale_features == 'normalize':
		features_mip = preprocessing.normalize(features_mip)
	elif scale_features == 'robust':
		features_mip = preprocessing.robust_scale(features_mip)



	###################################################################
	#Section 2A: PCA, Find the number of dimensions to map to
	###################################################################
	# Below, "information" is defined as (sum of variance of components)/(total variance)
	# Information is a scalar in [0,1]

	n_components=int(ceil(0.5*len(features_mip[0])))
	dim_70=0
	dim_90=0
	dim_95=0
	total_components=len(features_mip[0])

	pca = PCA(total_components)
	pca.fit(features_mip)
	proj = pca.transform(features_mip)


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
					'will be saved to text for the reduced feature space and reduced PCA feature space unless otherwise specified' %(line,n, n))
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
	plt.savefig('%s/PCA_information_%s.pdf' %(images_dir,stamp), bbox_inches='tight', pad_inches=0)
	if show == True:
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
		plt.savefig('%s/PCA C%s_%s.pdf' %(images_dir,(k+1),stamp), bbox_inches='tight', pad_inches=0)
		if show == True:
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
	plt.savefig('%s/PCA C1 & C2_%s.pdf' %(images_dir,stamp), bbox_inches='tight', pad_inches=0)
	if show == True:
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
		plt.savefig('%s/PCA C%s (names)_%s.pdf' %(images_dir,(k+1),stamp), bbox_inches='tight', pad_inches=0)
		if show == True:
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
	plt.savefig('%s/PCA C1 & C2 (names)_%s.pdf' %(images_dir,stamp), bbox_inches='tight', pad_inches=0)
	if show == True:
		plt.show()
	plt.close()


	###################################################################
	#Section 2C: PCA, Stats & save info
	###################################################################


	# Save the PCA components to text file
	# dim_95 is set automatically to keep 95% of information
	# dim_95 is automatically calculated from above

	DATA=[]
	header = ['name',]
	header.extend(['PCA %s' %(i+1) for i in range(total_components)])
	DATA.append(header)

	for line in instances_mip:
		DATA.append([line])

	components = proj[:,:total_components]

	for j in range(len(components)):
		DATA[j+1].extend(components[j])

	with open('%s/%s-full-PCA.csv' %(data_dir,stamp),'w') as f:
		writer = csv.writer(f)
		writer.writerows(DATA)

	DATA=[]
	header = ['name',]
	header.extend(['PCA %s' %(i+1) for i in range(dim_95)])
	DATA.append(header)

	for line in instances_mip:
		DATA.append([line])

	components = proj[:,:dim_95]

	for j in range(len(components)):
		DATA[j+1].extend(components[j])

	with open('%s/%s-reduced-PCA.csv' %(data_dir,stamp),'w') as f:
		writer = csv.writer(f)
		writer.writerows(DATA)




	# Grab highest weighted variables in PCA
	# Here, the features are grabbed that have the highest total abs(weight) over all components
	# Num features is arbitrarily set to grab dim_95 features
	pca_eigenvectors = [abs(pca.components_[i]) for i in range(n_components)]
	total_weights=[sum(x) for x in zip(*pca_eigenvectors)]
	total_weights=np.array(total_weights)
	indices = (-total_weights).argsort()[:dim_95]


	top_features =[str(data[0][index+1]) for index in indices]
	np.savetxt('%s/pca_top_features_%s.txt' %(results_dir,stamp), top_features, fmt='%s')

	DATA=[]
	header = ['name',]
	header.extend([i for i in top_features])
	DATA.append(header)


	for line in instances_mip:
		DATA.append([line])

	a=len(instances_mip)
	for j in range(len(data[0])):
		for feature in top_features:
			if data[0][j] == feature:
				for k in range(a):
					DATA[k+1].extend([data[k+1][j]])


	with open('%s/%s_reduced-byPCA.csv' %(data_dir,stamp),'w') as f:
		writer = csv.writer(f)
		writer.writerows(DATA)


	# Here are some statistics about the PCA
	# Print this info or save to file if relevant
	pca.explained_variance_ratio_
	pca.explained_variance_
	pca.mean_
	pca.n_components_
	pca.noise_variance_













































































## Haiku ##
# A voice from heaven
# Eats just Polish Pierogies
# Not a cartoon, Bart
# -- "Bart" by Alex

## Haiku ##
# Flannel and
# A social one loves to MIP
# David from Davis
# -- "David" by Alex

## Haiku ##
# C++ experts
# Gorana and Radovan
# Love milk for lunches
# -- "Gorana" by Alex
