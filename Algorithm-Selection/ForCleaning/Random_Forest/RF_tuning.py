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
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from collections import OrderedDict
from sklearn.preprocessing import RobustScaler
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor


# ex. of how to run code:
# python RF_tuning.py mipdev-features.csv performance.csv
# each row in mipdev-features.csv is an instance, the columns are features
# each row in performance.csv is an instance, the columns are performance values for various solvers
# questions can be sent to Alex Georges: ageorges@ucsd.edu

def weight_producer(data):
	weights=[[1/(abs(i)+1) for i in line] for line in data]
	return weights

###################################################################

# Section 1A: Grabs Data

###################################################################

FEATURES=sys.argv[1]
PERFORMANCE=sys.argv[2]

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
weights = weight_producer(performances_tot)

## Choices for preprocessing the data
# normalize = scale to unit norm
# maxabs_scale = scale to [-1,1]
# scale = zero mean scaled to std one

features_tot = preprocessing.scale(features_tot)
#performances_tot = preprocessing.maxabs_scale(performances_tot, axis=1) 

###################################################################

# Section 2: Split into training/test sets
# may need to tune max_depth to avoid overfitting
# may need to tune parameters to achieve best results

###################################################################

X_train, X_test, y_train, y_test, indices_train, indices_test, weights_train, weights_test = train_test_split(features_tot, 
	performances_tot, instances_matched, weights, train_size=0.9,random_state=0)




###################################################################

# Section 3A: This section determines best function for determining
# max_features and best numer of trees in the forest.
# "Best" is defined as lowest error=1-R^2

# Author: Kian Ho <hui.kian.ho@gmail.com>
#         Gilles Louppe <g.louppe@gmail.com>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 Clause

###################################################################

RANDOM_STATE = 0


'''

ensemble_clfs = [
    ("RandomForestRegressor, max_features='sqrt'",
        RandomForestRegressor(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestRegressor, max_features='log2'",
        RandomForestRegressor(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestRegressor, max_features=None",
        RandomForestRegressor(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
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
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.savefig('RF_tune_ntrees_regress.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()


###################################################################

# Section 3B: This section determines best function for determining max_depth
# "Best" is defined as lowest error=1-R^2
# This is modeled on the code above
# This section is mainly useful if dealing with noisy data. 
# Otherwise, a deeper tree is best.

###################################################################

ensemble_clfs = [
    ("RandomForestRegressor, max_features='sqrt'",
        RandomForestRegressor(warm_start=False, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestRegressor, max_features='log2'",
        RandomForestRegressor(warm_start=False, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestRegressor, max_features=None",
        RandomForestRegressor(warm_start=False, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
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
plt.xlabel("max_depth")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.savefig('RF_tune_depth_regress.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()

'''
###################################################################

# Section 3C: 

###################################################################


a=len(instances_matched)
labels_tot = []
performances_tot=np.array(performances_tot)
for i in range(a):
	labels_tot.extend((performances_tot[i]).argsort()[:1])

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features_tot, 
	labels_tot, instances_matched, train_size=0.8,random_state=0)


ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 15
max_estimators = 550
min_depth = 2
max_depth = 50

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
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.savefig('RF_tune_ntrees_class.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()


###################################################################

# Section 3D: 

###################################################################



ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=False, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=False, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=False, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
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
plt.xlabel("max_depth")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.savefig('RF_tune_depth_regress.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()





'''







###################################################################

# Section 4: 

###################################################################
def sigmoid(x):
	return 1/(1+exp(-x))

sums = [sum(line) for line in weights_train]
sums = sums/max(sums)

sums = [sigmoid(x) for x in sums]
weights_train=np.array(weights_train)

y_train=np.array(y_train)

rf = RandomForestRegressor(max_features="auto",random_state=0, oob_score=True, max_depth=None, n_estimators=250, min_weight_fraction_leaf=0)

rf.fit(X_train, y_train, sample_weight=None)

y_rf = rf.predict(X_test)


y_test=np.array(y_test)


params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2, 'learning_rate': 0.05, 'loss': 'lad', 'verbose': 1}
boost = GradientBoostingRegressor(**params)
rf_boost = MultiOutputRegressor(boost)
rf_boost.fit(X_train,y_train)
y_bf=predict(X_test)


# Plot the results
fig, ax = plt.subplots()
plt.plot(y_test[:,0],y_bf[:,0], 'ro')
plt.title("Performance Data vs Predicted Performance")
plt.axis('tight')
plt.savefig('RF_performance.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()


# Check statistics of fitting test vs predicted to line
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test[:,0],y_bf[:,0])
print("r-squared:", r_value**2)
print("error:", std_err)




###################################################################
###################################################################
###################################################################



'''


























