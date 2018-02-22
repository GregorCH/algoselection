from numpy import *
import matplotlib.pyplot as plt
import os, string, sys
import numpy as np
import math
import csv
import numbers
import _pickle as serializer
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib.patches import FancyArrowPatch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
from sklearn.metrics import euclidean_distances
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats.mstats import gmean

from . import utilities
from .. import config, logger

log = logger._Logger.get_logger(__name__) # set module name for logging
cfg = config.global_config

# ex. of how to run code:

# random_forest_regressor_tuner(mipdev-features.csv, performance.csv)

# each row in mipdev-features.csv is an instance, the columns are features
# column zero in the features file must be the instance name
# each row in performance.csv is an instance, the columns are performance values for various solvers
# column zero in the performance file must be the instance name

## Sometimes the abbreviation will be made: 'rfr' = random forest regressor

# questions can be sent to Alex Georges: ageorges@ucsd.edu

def tune(
	FEATURES,
	PERFORMANCE,
	show = False,
	tune_trees = True,
	tune_depth = True,
	linear_fit = True,
	random_state = 0,
	scale_features = 'maxabs',
	scale_performance = None,
	images_dir = cfg.models_images_dir,
	data_dir = cfg.models_data_dir,
	results_dir = cfg.models_results_dir,
	test_dir = cfg.models_test_dir):

	'''This function is used to tune various parameters of the Random Forest Regressor.
	This should be run 1st in the workflow of the Random Forest Regressor, 2nd
	should be test and 3rd should be random_forest_regressor_predictor. The tuner
	will find certain optimal parameters.  This scanning assumes error can be tuned
	over each parameter independently.

	:param FEATURES: Features file. This should be in CSV format, with column 0
	 				 being the instance name and row zero being the names of the
					 features.
	:param PERFORMANCE: Performance file. This should be in CSV format, with
						column 0 being the instance name and row zero being the
						names of the various solvers.
	:param show: Whether to show images as they are being produced. default=False
	:param tune_trees: Whether to tune the number of trees in the forest. default=True
	:param tune_depth: Whether to tune the depth of trees in the forest. default=True
	:param linear_fit: Whether to perform a linear fit analysis. default=True
	:param random_state: Specify the random seed (int) to be used in training the
	 					 Random Forest. default=0
	:param scale_features: There are various ways to scale the features data. The
	 					   scaling is done column-wise (i.e. on each feature individually).
						   default='maxabs'.
						   Options:

						   - maxabs = Scale to [-1,1]
						   - scale = Zero mean and unit stdev
						   - minmax = Translate and scale to [0,1]
						   - normalize = Normalize each feature to unit norm
						   - robust = Shift outliers in according to interquartile range
	:param scale_performance: There are various ways to scale the performance data.
	  						  The scaling is done row-wise (i.e. on each instance
							  individually).  default=None.
							  Options:

							  - maxabs = Scale to [-1,1]
							  - scale = Zero mean and unit stdev
							  - minmax = Translate and scale to [0,1]
							  - normalize = Normalize each row to unit norm
							  - default_scale = Add 1000 to each entry, and row-wise divide by default performance
	:param images_dir: Directory to dump images.
	:param data_dir: Directory to dump data.
	:param results_dir: Directory to dump results.
	:param test_dir: Directory to dump test results.
	:return: Various plots. The filename of every saved output automatically has
	 		 the input file names used to produce it.
			 Plots all returned in PDF format and are automatically saved:

			 - Plot 1: Relative error vs number of trees for testing data.
			 		   Relative error = :math:`1-R^2`. Produced if tune_trees=True.
			 - Plot 2: Relative errors vs max depth for testing data.
			 		   Relative error = :math:`1-R^2`.  Produced if tune_depth=True.
			 - Plot 3: Actual performance (x-axis) vs predicted performance (y-axis)
			 		   and the correlation between these computed using the Pearson
					   Correlation. Since this value is sensitive to the random
					   state, this value is automatically averaged over 100 different
					   linear fits. A higher value for the Pearson Correlation will
					   result in better prediction performance in the end, so this
					   is a very useful tool. Produced if linear_fit=True.
	'''

	###################################################################
	# Section 1A: Grabs Data
	###################################################################

	stamp = '%s_%s' %(os.path.basename(FEATURES).split('.')[0],os.path.basename(PERFORMANCE).split('.')[0])

	with open("%s" %(FEATURES)) as f:
		reader = csv.reader(f, delimiter=",")
		data = list(reader)
	#instances = [os.path.basename(line[0]).split('.')[0] for line in data[1:]]
	features = [line for line in data[1:]]
	feature_names = [line for line in data[0]]

	with open("%s" %(PERFORMANCE)) as f:
		reader = csv.reader(f, delimiter=",")
		data = list(reader)
	performances = [line for line in data[1:]]

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
	# Section 2: Split into training/test sets
	# may need to tune max_depth to avoid overfitting
	# may need to tune parameters to achieve best results
	###################################################################

	X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features_tot,
		performances_tot, instances_matched, train_size=0.9,random_state=random_state)


	###################################################################
	# Section 3A: This section determines best function for determining
	# max_features and best numer of trees in the forest.
	# "Best" is defined as lowest error=1-R^2

	# Author: Kian Ho <hui.kian.ho@gmail.com>
	#         Gilles Louppe <g.louppe@gmail.com>
	#         Andreas Mueller <amueller@ais.uni-bonn.de>
	# License: BSD 3 Clause
	###################################################################
	if tune_trees == True:

		ensemble_clfs = [
		    ("RandomForestRegressor, max_features='sqrt'",
		        RandomForestRegressor(warm_start=True, oob_score=True,
		                               max_features="sqrt",
		                               random_state=random_state)),
		    ("RandomForestRegressor, max_features='log2'",
		        RandomForestRegressor(warm_start=True, max_features='log2',
		                               oob_score=True,
		                               random_state=random_state)),
		    ("RandomForestRegressor, max_features=None",
		        RandomForestRegressor(warm_start=True, max_features=None,
		                               oob_score=True,
		                               random_state=random_state))
		]

		# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
		error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

		# Range of `n_estimators` values to explore.
		min_estimators = 15
		max_estimators = 550

		for label, clf in ensemble_clfs:
		    for i in range(min_estimators, max_estimators + 1):
		        clf.set_params(n_estimators=i)
		        clf.fit(X_train, y_train)

		        # Record the OOB error for each `n_estimators=i` setting.
		        oob_error = 1 - clf.oob_score_
		        error_rate[label].append((i, oob_error))

		# Generate the "OOB error rate" vs. "n_estimators" plot.
		for label, clf_err in error_rate.items():
		    xs, ys = zip(*clf_err)
		    plt.plot(xs, ys, label=label)

		plt.xlim(min_estimators, max_estimators)
		plt.xlabel("N Estimators")
		plt.ylabel("Relative Error")
		plt.title("Relative Test Error vs N Estimators")
		plt.legend(loc="upper right")
		plt.savefig('%s/rfr_tune_ntrees_%s.pdf' %(images_dir,stamp), bbox_inches='tight', pad_inches=0)  ## 'rfr' = random forest regressor
		if show == True:
			plt.show()
		plt.close()


	###################################################################
	# Section 3B: This section determines best function for determining max_depth
	# "Best" is defined as lowest error=1-R^2
	# This is modeled on the code above
	# This section is mainly useful if dealing with noisy data.
	# Otherwise, a deeper tree is best.
	###################################################################
	if tune_depth == True:

		ensemble_clfs = [
		    ("RandomForestRegressor, max_features='sqrt'",
		        RandomForestRegressor(warm_start=False, oob_score=True,
		                               max_features="sqrt",
		                               random_state=random_state)),
		    ("RandomForestRegressor, max_features='log2'",
		        RandomForestRegressor(warm_start=False, max_features='log2',
		                               oob_score=True,
		                               random_state=random_state)),
		    ("RandomForestRegressor, max_features=None",
		        RandomForestRegressor(warm_start=False, max_features=None,
		                               oob_score=True,
		                               random_state=random_state))
		]

		# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
		error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

		# Range of `max_depth` values to explore.
		min_depth = 2
		max_depth = 50

		for label, clf in ensemble_clfs:
		    for i in range(min_depth, max_depth + 1):
		        clf.set_params(max_depth=i)
		        clf.fit(X_train, y_train)

		        # Record the OOB error for each `n_estimators=i` setting.
		        oob_error = 1 - clf.oob_score_
		        error_rate[label].append((i, oob_error))

		# Generate the "OOB error rate" vs. "max_depth" plot.
		for label, clf_err in error_rate.items():
		    xs, ys = zip(*clf_err)
		    plt.plot(xs, ys, label=label)

		plt.xlim(min_depth, max_depth)
		plt.xlabel("Max Depth")
		plt.ylabel("Relative Error")
		plt.title("Relative Test Error vs Max Depth")
		plt.legend(loc="upper right")
		plt.savefig('%s/rfr_tune_depth_%s.pdf' %(images_dir, stamp), bbox_inches='tight', pad_inches=0)  ## 'rfr' = random forest regressor
		if show == True:
			plt.show()
		plt.close()

	###################################################################
	# Section 4: Do magic
	###################################################################
	if linear_fit == True:


		# Get some better statistics by doing averaging over 50 seeds
		R_values=[]

		for i in range(100):
			random_state = int(i)

			X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(features_tot,
				performances_tot, instances_matched, train_size=0.8,random_state=random_state)

			rf = RandomForestRegressor(max_features="sqrt",random_state=random_state, oob_score=True, max_depth=None, n_estimators=250)
			rf.fit(X_train, y_train)
			y_predict = rf.predict(X_test)
			y_test=np.array(y_test)

			x=y_test[:,1]
			y=y_predict[:,1]

			slope, intercept, r_value, p_value, std_err = linregress(x,y)
			R_values.append(r_value)

		mean_r=mean(R_values)


		rf = RandomForestRegressor(max_features="sqrt",random_state=random_state, oob_score=True, max_depth=None, n_estimators=250)
		rf.fit(X_train, y_train)
		y_predict = rf.predict(X_test)
		y_test=np.array(y_test)


		# Plot the results
		# Check statistics of fitting predicted vs actual to line
		# R^2 = 1 means a linear model is a good fit.
		# r value and pearson correlation coefficient are used interchangeable since they are mathematically equivalent in this case.
		# Any line with positive slope is good for our purposes.

		x=y_test[:,1]
		y=y_predict[:,1]

		slope, intercept, r_value, p_value, std_err = linregress(x,y)
		fit = intercept+slope*x

		fig, ax = plt.subplots()
		plt.plot(x,y, 'ro')
		plt.title("Predicted vs Actual Performance for Solver emphasis_easycip_PDInt")
		plt.axis('tight')
		plt.xlabel('Actual Performance')
		plt.ylabel('Predicted Performance')
		plt.plot(x, fit, 'b', label='mean correlation=%0.3f' %(mean_r))
		plt.legend()
		plt.savefig('%s/rfr_performance_%s.pdf' %(images_dir,stamp), bbox_inches='tight', pad_inches=0)
		if show == True:
			plt.show()
		plt.close()
		print("Mean pearson correlation coefficient:", mean_r)

######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

def train(
	FEATURES,
	PERFORMANCE,
	FEATURES2,
	random_state = 0,
	boost = False,
	reduce = False,
	scale_features = 'maxabs',
	scale_performance = None,
	max_features = 'sqrt',
	n_estimators = 250,
	max_depth = None,
	images_dir = cfg.models_images_dir,
	data_dir = cfg.models_data_dir,
	results_dir = cfg.models_results_dir,
	test_dir = cfg.models_test_dir):

	'''This function is used to make predictions on which solvers to use for a
	given instance.  This requires training feature and performance data, as well
	as feature data for the instances for which these predictions are being made.
	This should be run 3rd in the workflow of the Random Forest Regressor, 1st
	should be the random_forest_tuner and 2nd should be test.

	:param FEATURES: Features file. This should be in CSV format, with column 0
	 				 being the instance name and row zero being the names of the
					 features.
	:param PERFORMANCE: Performance file. This should be in CSV format, with
						column 0 being the instance name and row zero being the
						names of the various solvers.
	:param FEATURES2: Features file of instances to make predictions on. This
					should be in CSV format, with column 0 being the instance
					name and row zero being the names of the features.
	:param random_state: Specify the random seed (int) to be used in training
						 the Random Forest. default=0
	:param boost: Whether to use AdaBoost in the portfolio prediction.
	:param reduce: Whether to train on a subset of the input features file. This
	 			   subset is chosen to be the most important features by the
				   random forest.
	:param scale_features: There are various ways to scale the features data. The
						   scaling is done column-wise (i.e. on each feature
						   individually).  default='maxabs'.
						   Options:

						   - maxabs = Scale to [-1,1]
						   - scale = Zero mean and unit stdev
						   - minmax = Translate and scale to [0,1]
						   - normalize = Normalize each feature to unit norm
						   - robust = Shift outliers in according to interquartile range
	:param scale_performance: There are various ways to scale the performance data.
							  The scaling is done row-wise (i.e. on each instance
							  individually). default=None.
							  Options:

							  - maxabs = Scale to [-1,1]
							  - scale = Zero mean and unit stdev
							  - minmax = Translate and scale to [0,1]
							  - normalize = Normalize each row to unit norm
							  - default_scale = Add 1000 to each entry, and
							  	row-wise divide by default performance
	:param max_features: The number of features to consider for splitting at each
	 					 node in the decision tree. This should be optimized by
						 the user with the tune function.
	:param n_estimators: The number of trees to use in the decision forest. This
	 					 should be optimized by the user with the tune function.
	:param max_depth: The max depth to train each tree in the forest to. This
					  should be optimized by the user with the tune function.
	:param images_dir: Directory to dump images.
	:param data_dir: Directory to dump data.
	:param results_dir: Directory to dump results.
	:param test_dir: Directory to dump test results.
	:return: A list of predictions of which solvers to use for each instance. The
	 		 filename of every saved output automatically has the input file names
			 used to produce it. The output file follows this convention:
			 Column 0 = instance name
			 Column 1 = predicted best solver
			 Column 2 = predicted 2nd best solver
			 Column 3 = predicted 3rd best solver
			 Column 4 = etc...
	'''

	###################################################################
	# Section 1A: Grabs Data
	###################################################################
	stamp = '%s_%s' %(os.path.basename(FEATURES).split('.')[0],os.path.basename(PERFORMANCE).split('.')[0])

	with open("%s" %(FEATURES)) as f:
		reader = csv.reader(f, delimiter=",")
		data = list(reader)
	features = [line for line in data[1:]] #does include instance name as 0th column
	feature_names = [line for line in data[0]]

	with open("%s" %(PERFORMANCE)) as f:
		reader = csv.reader(f, delimiter=",")
		data = list(reader)
	performances = [line for line in data[1:]]
	solvers = [line for line in data[0][1:]]

	with open("%s" %(FEATURES2)) as f:
		reader = csv.reader(f, delimiter=",")
		data = list(reader)
	features_new = [line[1:] for line in data[1:]] #does not include instance name as 0th column
	instances_new = [os.path.basename(line[0]).split('.')[0] for line in data[1:]]

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
	features_tot = [line[1:] for line in features_matched]
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
		features_new = preprocessing.scale(features_new)
	elif scale_features == 'maxabs':
		features_tot = preprocessing.maxabs_scale(features_tot)
		features_new = preprocessing.maxabs_scale(features_new)
	elif scale_features == 'minmax':
		features_tot = preprocessing.minmax_scale(features_tot)
		features_new = preprocessing.minmax_scale(features_new)
	elif scale_features == 'normalize':
		features_tot = preprocessing.normalize(features_tot)
		features_new = preprocessing.normalize(features_new)
	elif scale_features == 'robust':
		features_tot = preprocessing.robust_scale(features_tot)
		features_new = preprocessing.robust_scale(features_new)

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
	# Section 2: Split into training/test sets
	# may need to tune max_depth to avoid overfitting
	# may need to tune parameters to achieve best results
	###################################################################

	X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(features_tot,
		performances_tot, instances_matched, train_size=0.999,random_state=random_state)

	y_test =[[float(i) for i in j] for j in y_test] # There's some string issue.  This converts to floats.
	y_train =[[float(i) for i in j] for j in y_train]

	###################################################################
	# Section 3A: Train up a RF regressor with parameters tuned according
	# to RF_tuning.py
	###################################################################
	solver_number=14

	if (boost==False) & (reduce==False):

		# Train up a Random Forest
		rf_regress = RandomForestRegressor(max_features=max_features,random_state=random_state, oob_score=True,
			max_depth=max_depth, n_estimators=n_estimators, verbose=1)
		rf_regress.fit(X_train, y_train)
		y_predict = rf_regress.predict(features_new)

		error_regress = 1 - rf_regress.oob_score_

		a=len(y_predict)
		portfolio = [[] for i in range(a)]
		for i in range(a):
			indices = (y_predict[i]).argsort()[:int(solver_number)]
			name = str(instances_new[i])
			portfolio[i] += name,
			for index in indices:
				portfolio[i] += solvers[index],

		with open('%s/rfr_%s.csv' %(results_dir,stamp),'w') as f:
			writer = csv.writer(f)
			writer.writerows(portfolio)

		utilities.save_classifier(rf_regress, 'rfregress')

	###################################################################
	# Section 3B: Do some feature selection on RF regressor
	###################################################################
	rf_regress = RandomForestRegressor(max_features=max_features,random_state=random_state, oob_score=True,
		max_depth=max_depth, n_estimators=n_estimators, verbose=0)
	selector = SelectFromModel(rf_regress)
	selector.fit(X_train, y_train)
	X_train_reduced = selector.transform(X_train)
	features_new_reduced = selector.transform(features_new)


	###################################################################
	# Section 3C: Train reduced RF regressor based on top features
	# predicted by RF regressor
	###################################################################

	if (boost==False) & (reduce==True):

		# Create a new random forest classifier for the most important features
		rf_regress_reduced = RandomForestRegressor(max_features=max_features,random_state=random_state, oob_score=True,
			max_depth=max_depth, n_estimators=n_estimators, verbose=1)
		rf_regress_reduced.fit(X_train_reduced, y_train)
		y_predict = rf_regress_reduced.predict(features_new_reduced)

		error_regress_reduced = 1 - rf_regress_reduced.oob_score_

		a=len(y_predict)
		portfolio = [[] for i in range(a)]
		for i in range(a):
			indices = (y_predict[i]).argsort()[:int(solver_number)]
			name = str(instances_new[i])
			portfolio[i] += name,
			for index in indices:
				portfolio[i] += solvers[index],

		with open('%s/rfr_reduced_%s.csv' %(results_dir,stamp),'w') as f:
			writer = csv.writer(f)
			writer.writerows(portfolio)

		utilities.save_classifier(rf_regress_reduced, 'rfregress_reduced')

	###################################################################
	# Section 3D: Train Gradient Boosted Regression Forest
	###################################################################
	params = {'n_estimators': 200, 'max_depth': max_depth, 'learning_rate': 0.01, 'loss': 'lad', 'verbose': 1, 'max_features': 'sqrt', 'random_state': random_state, 'subsample': 0.99}

	if (boost==True) & (reduce==False):

		boost_regress = GradientBoostingRegressor(**params)
		rf_regress_boost = MultiOutputRegressor(boost_regress)
		rf_regress_boost.fit(X_train,y_train)
		y_predict = rf_regress_boost.predict(features_new)

		a=len(y_predict)
		portfolio = [[] for i in range(a)]
		for i in range(a):
			indices = (y_predict[i]).argsort()[:int(solver_number)]
			name = str(instances_new[i])
			portfolio[i] += name,
			for index in indices:
				portfolio[i] += solvers[index],

		with open('%s/rfr_boosted_%s.csv' %(results_dir,stamp),'w') as f:
			writer = csv.writer(f)
			writer.writerows(portfolio)

		utilities.save_classifier(rf_regress_boost, 'rfregress_boost')

	if (boost==True) & (reduce==True):

		boost_regress_reduced = GradientBoostingRegressor(**params)
		rf_regress_boost_reduced = MultiOutputRegressor(boost_regress_reduced)
		rf_regress_boost_reduced.fit(X_train_reduced,y_train)
		y_predict = rf_regress_boost_reduced.predict(features_new_reduced)

		a=len(y_predict)
		portfolio = [[] for i in range(a)]
		for i in range(a):
			indices = (y_predict[i]).argsort()[:int(solver_number)]
			name = str(instances_new[i])
			portfolio[i] += name,
			for index in indices:
				portfolio[i] += solvers[index],

		with open('%s/rfr_reduced+boosted_%s.csv' %(results_dir,stamp),'w') as f:
			writer = csv.writer(f)
			writer.writerows(portfolio)

		utilities.save_classifier(rf_regress_boost_reduced, 'rfregress_boost_reduced')

######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

def predict(instance_name, feature_data, performance_data):
    # to load trained model use utilities.load_classifier
    # example: classifier = utilities.load_classifier('hydra_randomforest')
    # this will search for file named hydra_randomforest.model in directory
    # pointed by Config's models_dir parameter.
    # By default this is set to gripsPredictorPkg/data/models

	# suggest that feature_data and performance_data are of list type
	# (don't have better proposal right now)

    raise NotImplementedError('This method should use some of the trained models, \
        problem instance features and performance data to predict algorithm ranking')

######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

def test(
	FEATURES,
	PERFORMANCE,
	nseeds = 100,
	scale_features = 'maxabs',
	scale_performance = None,
	max_features = 'sqrt',
	n_estimators = 250,
	max_depth = None,
	images_dir = cfg.models_images_dir,
	data_dir = cfg.models_data_dir,
	results_dir = cfg.models_results_dir,
	test_dir = cfg.models_test_dir):

	'''The output predictions of the Random Forest Regressor are sensitive to the
	random state used to split the data into training/test sets and used in the
	Random Forest Regressor itself.  To construct an accurate representation of the
	performance of the predictor, various portfolio predictions should be averaged
	over this random state.

	:param FEATURES: Features file. This should be in CSV format, with column 0
					 being the instance name and row zero being the names of the
					 features.
	:param PERFORMANCE: Performance file. This should be in CSV format, with
						column 0 being the instance name and row zero being the
						names of the various solvers.
	:param nseeds: Number of seeds to use to produce various results which will
				   be averaged over. default=100
	:param scale_features: There are various ways to scale the features data. The
						   scaling is done column-wise (i.e. on each feature
						   individually). default='maxabs'.
						   - maxabs = Scale to [-1,1]
						   - scale = Zero mean and unit stdev
						   - minmax = Translate and scale to [0,1]
						   - normalize = Normalize each feature to unit norm
						   - robust = Shift outliers in according to interquartile range
	:param scale_performance: There are various ways to scale the performance data.
							  The scaling is done row-wise (i.e. on each instance
							  individually).  default=None.
							  - maxabs = Scale to [-1,1]
							  - scale = Zero mean and unit stdev
							  - minmax = Translate and scale to [0,1]
							  - normalize = Normalize each row to unit norm
							  - default_scale = Add 1000 to each entry, and row-wise divide by default performance
	:param max_features: The number of features to consider for splitting at each
	 					 node in the decision tree. This should be optimized by
						 the user with the tune function.
	:param n_estimators: The number of trees to use in the decision forest. This
						 should be optimized by the user with the tune function.
	:param max_depth: The max depth to train each tree in the forest to. This
					  should be optimized by the user with the tune function.
	:param images_dir: Directory to dump images.
	:param data_dir: Directory to dump data.
	:param results_dir: Directory to dump results.
	:param test_dir: Directory to dump test results.
	:return: For each seed and for each prediction method, return a list of
			 predictions of which solvers to use for each instance. The filename
			 of every saved output automatically has the input file names used
			 to produce it. The output files follows this convention:
			 Column 0 = instance name
			 Column 1 = predicted best solver
			 Column 2 = predicted 2nd best solver
			 Column 3 = predicted 3rd best solver
			 Column 4 = etc...
			 Four methods are used to produce portfolio predictions:
			 1) Random Forest Regressor
			 2) Reduced Random Forest Regressor
			 3) Boosted Random Forest Regressor
			 4) Reduced and Boosted Random Forest Regressor
			 The Reduced regressors use a subset of the input features which it determines is the most important subset.
			 The boosted regressors use AdaBoost.
	'''


	###################################################################
	# Section 1A: Grabs Data
	###################################################################

	with open("%s" %(FEATURES)) as f:
		reader = csv.reader(f, delimiter=",")
		data = list(reader)
	features = [line for line in data[1:]] #does include instance name as 0th column
	feature_names = [line for line in data[0]]

	with open("%s" %(PERFORMANCE)) as f:
		reader = csv.reader(f, delimiter=",")
		data = list(reader)
	performances = [line for line in data[1:]]
	solvers = [line for line in data[0][1:]]


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
		for line in performances:
			if instance==line[0]:
				performances_matched.append(line)

	performances_tot = [line[1:] for line in performances_matched]
	features_tot = [line[1:] for line in features_matched]

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
	# Section 2: Split into training/test sets
	# may need to tune max_depth to avoid overfitting
	# may need to tune parameters to achieve best results
	# Iterate over seeds to reduce statistical fluctuations
	###################################################################

	solver_number = 14
	for i in range(int(nseeds)):
		random_state = int(i)

		stamp = '%s_%s_s%s' %(os.path.basename(FEATURES).split('.')[0],os.path.basename(PERFORMANCE).split('.')[0],str(random_state))

		X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(features_tot,
			performances_tot, instances_matched, train_size=0.8,random_state=random_state)

		y_test =[[float(i) for i in j] for j in y_test] # There's some string issue.  This converts to floats.
		y_train =[[float(i) for i in j] for j in y_train]


		###################################################################
		# Section 3A: Train up a RF regressor with parameters tuned according
		# to RF_tuning.py
		###################################################################

		# Train up a Random Forest
		rf_regress = RandomForestRegressor(max_features=max_features,random_state=random_state, oob_score=True,
			max_depth=max_depth, n_estimators=n_estimators, verbose=1)

		rf_regress.fit(X_train, y_train)
		y_predict = rf_regress.predict(X_test)

		error_regress = 1 - rf_regress.oob_score_
		MAE_regress = mean_absolute_error(y_test,y_predict)

		a=len(y_predict)
		portfolio = [[] for i in range(a)]
		for i in range(a):
			indices = (y_predict[i]).argsort()[:int(solver_number)]
			name = str(names_test[i])
			portfolio[i] += name,
			for index in indices:
				portfolio[i] += solvers[index],

		with open('%s/rfr_%s.csv' %(test_dir,stamp),'w') as f:
			writer = csv.writer(f)
			writer.writerows(portfolio)


		###################################################################
		# Section 3B: Do some feature selection on RF regressor
		###################################################################

		# Feature Selection
		selector = SelectFromModel(rf_regress)
		selector.fit(X_train, y_train)

		# X_reduced has instances as rows and reduced feature space as columns
		X_train_reduced = selector.transform(X_train)
		X_test_reduced = selector.transform(X_test)


		###################################################################
		# Section 3C: Train reduced RF regressor based on top features
		# predicted by RF regressor
		###################################################################

		# Create a new random forest classifier for the most important features
		rf_regress_reduced = RandomForestRegressor(max_features=max_features,random_state=random_state, oob_score=True,
			max_depth=max_depth, n_estimators=n_estimators, verbose=1)
		rf_regress_reduced.fit(X_train_reduced, y_train)
		y_predict = rf_regress_reduced.predict(X_test_reduced)

		error_regress_reduced = 1 - rf_regress_reduced.oob_score_
		MAE_regress_reduced = mean_absolute_error(y_test,y_predict)

		a=len(y_predict)
		portfolio = [[] for i in range(a)]
		for i in range(a):
			indices = (y_predict[i]).argsort()[:int(solver_number)]
			name = str(names_test[i])
			portfolio[i] += name,
			for index in indices:
				portfolio[i] += solvers[index],

		with open('%s/rfr_reduced_%s.csv' %(test_dir,stamp),'w') as f:
			writer = csv.writer(f)
			writer.writerows(portfolio)


		###################################################################
		# Section 3D: Train Gradient Boosted Regression Forest
		###################################################################

		params = {'n_estimators': 200, 'max_depth': max_depth, 'learning_rate': 0.01, 'loss': 'lad', 'verbose': 1, 'max_features': 'sqrt', 'random_state': random_state, 'subsample': 0.99}

		boost_regress = GradientBoostingRegressor(**params)
		rf_regress_boost = MultiOutputRegressor(boost_regress)
		rf_regress_boost.fit(X_train,y_train)
		y_predict = rf_regress_boost.predict(X_test)
		MAE_regress_boost = mean_absolute_error(y_test,y_predict)


		a=len(y_predict)
		portfolio = [[] for i in range(a)]
		for i in range(a):
			indices = (y_predict[i]).argsort()[:int(solver_number)]
			name = str(names_test[i])
			portfolio[i] += name,
			for index in indices:
				portfolio[i] += solvers[index],

		with open('%s/rfr_boosted_%s.csv' %(test_dir,stamp),'w') as f:
			writer = csv.writer(f)
			writer.writerows(portfolio)

		boost_regress_reduced = GradientBoostingRegressor(**params)
		rf_regress_boost_reduced = MultiOutputRegressor(boost_regress_reduced)
		rf_regress_boost_reduced.fit(X_train_reduced,y_train)
		y_predict = rf_regress_boost_reduced.predict(X_test_reduced)
		MAE_regress_boost_reduced = mean_absolute_error(y_test,y_predict)

		a=len(y_predict)
		portfolio = [[] for i in range(a)]
		for i in range(a):
			indices = (y_predict[i]).argsort()[:int(solver_number)]
			name = str(names_test[i])
			portfolio[i] += name,
			for index in indices:
				portfolio[i] += solvers[index],

		with open('%s/rfr_reduced+boosted_%s.csv' %(test_dir,stamp),'w') as f:
			writer = csv.writer(f)
			writer.writerows(portfolio)


	###################################################################
	# Section 5: Print important stats to screen
	###################################################################
	# print()
	# print('The Random Forest predicts the dimension of features to train on is around %s' %(n_features_class))
	# print()
	# #print('The test percent-error for the Full Random Forest is = %.4f' %(error))
	# #print('The test percent-error for the Reduced Random Forest is = %.4f' %(error_reduced))
	# print()
	# print('MAE Regression Forest:                         %0.3f, %0.3f' %(MAE_regress,error_regress))
	# print('MAE Reduced Regression Forest:                 %0.3f, %0.3f' %(MAE_regress_reduced,error_regress_reduced))
	# print('MAE Boosted Regression Forest:                 %0.3f' %(MAE_regress_boost))
	# print('MAE Boosted Reduced Regression Forest:         %0.3f' %(MAE_regress_boost_reduced))
	# print()



























































































## Poem ##
  # "A planet doesn't explode of itself," said drily
  # The Martian astronomer, gazing off into the air --
  # "That they were able to do it is proof that highly
  # Intelligent beings must have been living there."
  # --"Earth" by John hall Wheelock
