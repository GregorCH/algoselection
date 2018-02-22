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

	'''Multidimensional scaling is a technique to map high dimeinsional data to
	lower dimensions. The heuristic used is to preserve pairwise distances as well
	as possible in this mapping.  In this function, it is used to:

	  1) Understand the dimensionality of the feature space according to MDS
	  2) Visualize how the data maps to 2D and 3D

	Further functionality can be added to more closely match the goals of the
	Principal Component Analysis function.

	:param FILE: Features file. This should be in CSV format, with column 0 being
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
	:param images_dir: Directory to dump images.  default='../images'
	:param data_dir: Directory to dump data.  default='../data'
	:param results_dir: Directory to dump results.  default='../results'
	:return: Plots all returned in PDF format.  Plots are automatically saved:

	     - Plot 1: Information retained with respect to dimension being mapped to
	       x-axis is scaled by heuristically.  Change this if plot isn't nice.
	     - Plot 2: MDS in d=2
	     - Plot 3: MDS in d=2, with names of instances attached to points
	     - Plot 4: MDS in d=3.  Plot is not automatically saved.

	'''

	###################################################################
	# Section 1: Grabs Feature Data
	###################################################################

	stamp = '%s' %(os.path.basename(FILE).split('.')[0])
	print(stamp)
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
	# Section 2A: MDS, Find the number of dimensions to map to
	# This can take a few minutes, so feel free to grab a coffee at this point
	###################################################################
	# Below, "information" is defined as (1-stress/scale)
	# The scale here is stress[dimension(2)]
	# Information is a scalar in [0,1]
	### Stress is defined as the sum of squared difference between
	### distances in the embedded space and distances in the original space

	max_dim=int(ceil(0.1*len(features_mip[0])))

	print()
	print("Max dimension projecting to is %s" %(max_dim))
	print()

	stress, dimension=[],[]
	fig, ax = plt.subplots()
	for i in range(2,max_dim+1): # choose the range of dimensions to map to
		print('Projecting to dimension %s' %i)
		mds = manifold.MDS(i) # number of dimensions to map to
		proj = mds.fit_transform(features_mip).T
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
	plt.savefig('%s/MDS_information_%s.pdf' %(images_dir,stamp), bbox_inches='tight', pad_inches=0)
	if show == True:
		plt.show()
	plt.close()

	###################################################################
	# Section 2B: MDS, Draw 2D and 3D MDS plots
	###################################################################

	print()
	print('Currently producing some more images...')

	# Draw 2D MDS
	mds = manifold.MDS(2) # number of dimensions to map to
	proj = mds.fit_transform(features_mip).T
	fig, ax = plt.subplots()
	plt.plot(proj[0],proj[1], 'ro')
	plt.title("Feature MDS")
	plt.xscale("symlog")
	plt.yscale("symlog")
	plt.axis('tight')
	plt.savefig('%s/MDS_%s.pdf' %(images_dir,stamp), bbox_inches='tight', pad_inches=0)
	if show == True:
		plt.show()
	plt.close()

	# Do the same as the above, just with names of instances attached to points
	proj = mds.fit_transform(features_mip).T
	fig, ax = plt.subplots()
	for i, txt in enumerate(instances_mip):
	   ax.annotate(txt, (proj[0][i],proj[1][i]))
	plt.plot(proj[0],proj[1], 'ro')
	plt.title("Feature MDS")
	plt.xscale("symlog")
	plt.yscale("symlog")
	plt.axis('tight')
	plt.savefig('%s/MDS (names)_%s.pdf' %(images_dir,stamp), bbox_inches='tight', pad_inches=0)
	if show == True:
		plt.show()
	plt.close()


	# Draw 3D MDS.  3D Images will not save automatically.
	mds = manifold.MDS(3) # number of dimensions to map to
	proj = mds.fit_transform(features_mip).T
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(proj[0],proj[1], proj[2])
	ax.set_xscale("symlog")
	ax.set_yscale("symlog")
	ax.set_zscale("symlog")
	if show == True:
		plt.show()
	plt.close()





	# The following code is broken.  It's supposed to label 3D MDS
	# mds = manifold.MDS(3) # number of dimensions to map to
	# proj = mds.fit_transform(features_mip).T
	# fig = figure()
	# ax = Axes3D
	# for i, txt in enumerate(instances_mip):
	# 	label = '%s' %(str(i))
	# 	ax.text(proj[0][i],proj[1][i],proj[2][i], label, size=20, zorder=1)
	# ax.scatter(proj[0],proj[1], proj[2])
	# ax.set_xscale("symlog")
	# ax.set_yscale("symlog")
	# ax.set_zscale("symlog")
	# plt.show()
	# plt.close()












































































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
