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
	FEATURES,
	PERFORMANCE,
	random_state = 0,
	scale_features = 'maxabs',
	scale_performance = None,
	images_dir = cfg.finvestig_images_dir,
	data_dir = cfg.finvestig_data_dir,
	results_dir = cfg.finvestig_results_dir):

	'''This function is used to produce a list of the top features as chosen by
	the random forest.  For each feature, the forest calculates a score. To construct
	the most important features, all features with a score above a threshold are
	chosen. Currently, the threshold is set to 2*mean, where the mean is the arithmetic
	mean taken over the various scores.

	:param FEATURES: Features file. This should be in CSV format, with column 0
	         being the instance name and row zero being the names of the
	         features.
	:param PERFORMANCE: Performance file. This should be in CSV format, with column
	         0 being the instance name and row zero being the names of the various solvers.
	:param random_state: Specify the random seed (int) to be used in training the
	         Random Forest. default=0
	:param scale_features: There are various ways to scale the features data. The
	             scaling is done column-wise (i.e. on each feature individually).
	             default='maxabs'.

	             - maxabs = Scale to [-1,1]
	             - scale = Zero mean and unit stdev
	             - minmax = Translate and scale to [0,1]
	             - normalize = Normalize each feature to unit norm
	             - robust = Shift outliers in according to interquartile range
	:param scale_performance: There are various ways to scale the performance data.
	              The scaling is done row-wise (i.e. on each instance individually).
	              default=None.

	              - maxabs = Scale to [-1,1]
	              - scale = Zero mean and unit stdev
	              - minmax = Translate and scale to [0,1]
	              - normalize = Normalize each row to unit norm
	              - default_scale = Add 1000 to each entry, and row-wise divide by default performance
	:param images_dir: Directory to dump images.
	:param data_dir: Directory to dump data.
	:param results_dir: Directory to dump results.
	:return: The filename of every saved output automatically has the input file
	     names used to produce it.
		 
	     - Text 1: Reduced by Random Forest Regressor space.  This is a subset
	       of the original Feature space, with most important features chosen as
	       the subset.  Most important is a heuristic chosen by the Random Forest.
	       Automatically saved to CSV format in data_dir.
	'''

	###################################################################
	# Section 1A: Grabs Data
	###################################################################

	stamp = '%s_%s' %(os.path.basename(FEATURES).split('.')[0],os.path.basename(PERFORMANCE).split('.')[0])

	with open("%s" %(FEATURES)) as f:
		reader = csv.reader(f, delimiter=",")
		data_f = list(reader)
	#instances = [os.path.basename(line[0]).split('.')[0] for line in data[1:]]
	features = [line for line in data_f[1:]]
	feature_names = [line for line in data_f[0]]

	with open("%s" %(PERFORMANCE)) as f:
		reader = csv.reader(f, delimiter=",")
		data_p = list(reader)
	performances = [line for line in data_p[1:]]

	###################################################################
	# Section 1B: Sync up data so that only instances with both
	# feature vectors and performance data is trained on
	###################################################################
	performances_matched=[]
	features_matched=[]
	instances_matched=[]

	for line in features:
		instance_name=os.path.basename(line[0]).split('.')[0]
		for line in performances:
			if line[0]==instance_name:
				instances_matched.append(instance_name)

	for instance in instances_matched:
		for line in features:
			if instance==os.path.basename(line[0]).split('.')[0]:
				features_matched.append(line)

	for instance in instances_matched:
		for line in performances:
			if instance==line[0]:
				performances_matched.append(line)

	performances_tot = [line[1:] for line in performances_matched]
	features_tot = [line[1:-1] for line in features_matched] #the -1 here removes the empty string coming from feature selection
	# There's some string issue.  The following converts to floats:
	performances_tot = [[float(i) for i in j] for j in performances_tot]
	performances_tot=np.array(performances_tot)

	###################################################################
	# Section 1C: Scale the feature/performance data
	###################################################################
	# normalize = scale to unit norm
	# maxabs_scale = scale to [-1,1]
	# scale = zero mean scaled to std one

	if scale_features == 'scale':
		features_tot = preprocessing.scale(features_tot)
	elif scale_features == 'maxabs':
		features_tot = preprocessing.maxabs_scale(features_tot)
	elif scale_features == 'minmax':
		features_tot = preprocessing.minmax_scale(features_tot)
	elif scale_features == 'normalize':
		features_tot = preprocessing.normalize(features_tot)
	elif scale_features == 'robust':
		features_tot = preprocessing.robust_scale(features_tot)

	if scale_performance == 'scale':
		performances_tot = preprocessing.scale(performances_tot, axis=1)
	elif scale_performance == 'maxabs':
		performances_tot = preprocessing.maxabs_scale(performances_tot, axis=1)
	elif scale_performance == 'minmax':
		performances_tot = preprocessing.minmax_scale(performances_tot, axis=1)
	elif scale_performance == 'normalize':
		performances_tot = preprocessing.normalize(performances_tot, axis=1)
	elif scale_performance == 'default_scale':
		performances_tot =[[(float(i)+1000)/(float(line[0])+1000) for i in line] for line in performances_tot]

	performances_tot=np.array(performances_tot)

	###################################################################
	# Section 2: Find the top features and save reduced feature file to txt
	###################################################################

	# Train up a Random Forest
	rf_regress = RandomForestRegressor(max_features="sqrt",random_state=random_state, max_depth=None, n_estimators=250, verbose=0)
	rf_regress.fit(features_tot, performances_tot)

	# Feature Selection
	selector = SelectFromModel(rf_regress, prefit=True,threshold='2*mean')
	Indices = selector.get_support(indices=True)
	top_features=[feature_names[index+1] for index in Indices]
	np.savetxt('%s/rfr_top_features_%s.txt' %(results_dir,stamp), top_features, fmt='%s')


	DATA=[]
	header = ['name',]
	header.extend([i for i in top_features])
	DATA.append(header)

	for line in instances_matched:
		DATA.append([line])

	a=len(instances_matched)
	for j in range(len(data_f[0])):
		for feature in top_features:
			if data_f[0][j] == feature:
				for k in range(a):
					DATA[k+1].extend([data_f[k+1][j]])

	with open('%s/%s_reduced-byRFR.csv' %(data_dir,stamp),'w') as f:
		writer = csv.writer(f)
		writer.writerows(DATA)












































































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
