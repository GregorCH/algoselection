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


# ex. of how to run code:

# feature_investigator_correlations(mipdev-features.csv,1,20, show=True)
# the first input is the feature file, the two integers are the range of features to look at.
# The default is to produce the correlations of everything.

# feature_investigator_mds(mipdev-features.csv, show=True)

# feature_investigator_pca(mipdev-features.csv, show=True)

# each row in mipdev-features is an instance, the columns are features
# column zero must be the instance name
# questions can be sent to Alex Georges: ageorges@ucsd.edu

def investigate(
	FILE,
	scale_features = 'maxabs',
	start = 1,
	end = None,
	show = False,
	images_dir = cfg.finvestig_images_dir,
	data_dir = cfg.finvestig_data_dir,
	results_dir = cfg.finvestig_results_dir):

	'''This produces a heatmap of the correlations between features.  The correlation
	strength is computed using Pearson Correlation Coefficients.

	:param FILE: Features file.  This should be in CSV format, with column 0 being the instance name
				 and row zero being the names of the features.
	:param scale_features: There are various ways to scale the features data.
						   The scaling is done column-wise (i.e. on each feature individually).  default='maxabs'.
						   - maxabs = Scale to [-1,1]
						   - scale = Zero mean and unit stdev
						   - minmax = Translate and scale to [0,1]
						   - normalize = Normalize each feature to unit norm
						   - robust = Shift outliers in according to interquartile range
	:param start: Starting feature to include in the correlations heatmap.  default=1
	:param end: Last feature to include in the correlations heatmap. default=None
	:param show: Whether to show images as they are being produced.  default=False
	:param images_dir: Directory to dump images.  default='../images'
	:param data_dir: Directory to dump data.  default='../data'
	:param results_dir: Directory to dump results.  default='../results'
	:return: Heatmap in PDF format.  Plot is automatically saved. The filename of
			 every saved output automatically has the input file names used to produce it.
	'''

	###################################################################
	# Section 1: Grabs Feature Data
	###################################################################

	stamp = '%s' %(os.path.basename(FILE).split('.')[0])

	with open("%s" %(FILE)) as f:
		reader = csv.reader(f, delimiter=",")
		data = list(reader)

	instances_mip = [os.path.basename(line[0]).split('.')[0] for line in data[1:]]
	features_mip = [line[start:end] for line in data[1:]]

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
	# Section 2A: Pearson Correlation Heatmap
	###################################################################

	corr=np.corrcoef(features_mip,rowvar=False)
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True
	f, ax = plt.subplots(figsize=(11, 9))
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
	            square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=data[0][start:end], yticklabels=data[0][start:end])
	plt.xticks(rotation=90)
	plt.tick_params(labelsize=6)
	plt.yticks(rotation=0)
	plt.title("Feature Pearson Correlation Heatmap")
	plt.savefig('%s/Correlation Heatmap_%s.pdf' %(images_dir,stamp), bbox_inches='tight', pad_inches=0)
	if show == True:
		plt.show()
































































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
