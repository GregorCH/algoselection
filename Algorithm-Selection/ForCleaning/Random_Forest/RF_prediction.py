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
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor




# ex. of how to run code:
# python RF_prediction.py mipdev-features.csv performance.csv mipdev-features-predict.csv
# each row in mipdev-features.csv is an instance, the columns are features
# each row in performance.csv is an instance, the columns are performance values for various solvers
# questions can be sent to Alex Georges: ageorges@ucsd.edu



solver_number = 1
RANDOM_STATE=None


###################################################################

# Section 1A: Grabs Data

###################################################################

FEATURES=sys.argv[1]
PERFORMANCE=sys.argv[2]
FEATURES2=sys.argv[3]

stamp = '_%s__%s' %(FEATURES.split('.')[0],PERFORMANCE.split('.')[0])

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
features_PREDICT = [line[1:-1] for line in data[1:-1]] #does not include instance name as 0th column
instances_PREDICT = [os.path.basename(line[0]).split('.')[0] for line in data[1:]]



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
features_tot = [line[1:-1] for line in features_matched] #the -1 here removes the empty string coming from feature selection

# Also, assign labels so RF classifier can be trained.
# Label for an instance = best performing solver
a=len(instances_matched)
labels_tot = []
performances_tot=np.array(performances_tot)
for i in range(a):
	labels_tot.extend((performances_tot[i]).argsort()[:1])


## Choices for preprocessing the data
# normalize = scale to unit norm
# maxabs_scale = scale to [-1,1]
# scale = zero mean scaled to std one
# robust_scale = center to median and component-wise scale outliers

features_tot = preprocessing.scale(features_tot)
features_PREDICT = preprocessing.scale(features_PREDICT)
#performances_tot = preprocessing.normalize(performances_tot, axis=1) 




###################################################################

# Section 2: Split into training/test sets
# may need to tune max_depth to avoid overfitting
# may need to tune parameters to achieve best results

###################################################################

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(features_tot, 
	performances_tot, instances_matched, train_size=0.8,random_state=RANDOM_STATE)

y_test =[[float(i) for i in j] for j in y_test] # There's some string issue.  This converts to floats.
y_train =[[float(i) for i in j] for j in y_train]


###################################################################

# Section 3A: Train up a RF regressor with parameters tuned according
# to RF_tuning.py

###################################################################

# Train up a Random Forest                                                                                 
rf_regress = RandomForestRegressor(max_features="auto",random_state=RANDOM_STATE, oob_score=True, max_depth=10, n_estimators=250, criterion="mse",verbose=1)
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

with open('rf_regress_portfolio_%s.csv' %(stamp),'w') as f:
	writer = csv.writer(f)
	writer.writerows(portfolio)


###################################################################

# Section 3B: Do some feature selection on RF regressor

###################################################################

# Feature Selection
selector = SelectFromModel(rf_regress)
selector.fit(X_train, y_train)
n_features = selector.transform(X_train).shape[1]
Indices = selector.get_support(indices=True)
top_features=[feature_names[index+1] for index in Indices]
np.savetxt('rf_regress_top_features_%s.txt' %(stamp), top_features, fmt='%s')

# X_reduced has instances as rows and reduced feature space as columns
X_train_reduced = selector.transform(X_train)
X_test_reduced = selector.transform(X_test)


###################################################################

# Section 3C: Train reduced RF regressor based on top features
# predicted by RF regressor

###################################################################

# Create a new random forest classifier for the most important features
rf_regress_reduced = RandomForestRegressor(max_features="auto",random_state=RANDOM_STATE, oob_score=True, max_depth=10, n_estimators=250, criterion="mse", verbose=1)
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

with open('rf_regress_reduced_portfolio_%s.csv' %(stamp),'w') as f:
	writer = csv.writer(f)
	writer.writerows(portfolio)


###################################################################

# Section 3D: Train Gradient Boosted Regression Forest 

###################################################################

params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2, 'learning_rate': 0.05, 'loss': 'lad', 'verbose': 1}

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

with open('rf_regress_boost_portfolio_%s.csv' %(stamp),'w') as f:
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

with open('rf_regress_boost_reduced_portfolio_%s.csv' %(stamp),'w') as f:
	writer = csv.writer(f)
	writer.writerows(portfolio)


###################################################################

# Section 3E: Train up a Random Forest classifier

###################################################################

solver_number = 1


X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(features_tot, labels_tot, instances_matched, train_size=0.8,random_state=RANDOM_STATE)

# Train up a Random Forest                                                                                 
rf_class = RandomForestClassifier(max_features="auto",random_state=RANDOM_STATE, oob_score=True, max_depth=10, n_estimators=250,verbose=1)
rf_class.fit(X_train, y_train)
y_predict = rf_class.predict(X_test)

error_class = 1 - rf_class.oob_score_
MAE_class = mean_absolute_error(y_test,y_predict)


a=len(y_predict)
portfolio = [[] for i in range(a)]
for i in range(a):
	index = y_predict[i]
	name = str(names_test[i])
	portfolio[i] += name,
	portfolio[i] += solvers[index],

with open('rf_class_portfolio_%s.csv' %(stamp),'w') as f:
	writer = csv.writer(f)
	writer.writerows(portfolio)

###################################################################

# Section 3F: Do some feature selection on RF classifier

###################################################################

selector = SelectFromModel(rf_class)
selector.fit(X_train, y_train)
n_features_class = selector.transform(X_train).shape[1]
Indices = selector.get_support(indices=True)
top_features=[feature_names[index+1] for index in Indices]
np.savetxt('rf_class_top_features_%s.txt' %(stamp), top_features, fmt='%s')

# X_reduced has instances as rows and reduced feature space as columns
X_train_reduced = selector.transform(X_train)
X_test_reduced = selector.transform(X_test)



###################################################################

# Section 3G: Train up a reduced Random Forest classifier

###################################################################

# Train up a Random Forest                                                                                 
rf_class_reduced = RandomForestClassifier(max_features="auto",random_state=RANDOM_STATE, oob_score=True, max_depth=10, n_estimators=250,verbose=1)
rf_class_reduced.fit(X_train_reduced, y_train)
y_predict = rf_class_reduced.predict(X_test_reduced)

error_class_reduced = 1 - rf_class_reduced.oob_score_
MAE_class_reduced = mean_absolute_error(y_test,y_predict)


a=len(y_predict)
portfolio = [[] for i in range(a)]
for i in range(a):
	index = y_predict[i]
	name = str(names_test[i])
	portfolio[i] += name,
	portfolio[i] += solvers[index],

with open('rf_class_reduced_portfolio_%s.csv' %(stamp),'w') as f:
	writer = csv.writer(f)
	writer.writerows(portfolio)

###################################################################

# Section 3H: Train Gradient Boosted Classification Forest

###################################################################

params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2, 'learning_rate': 0.05, 'loss': 'deviance', 'verbose': 1}

rf_class_boost = GradientBoostingClassifier(**params)
rf_class_boost.fit(X_train,y_train)
y_predict = rf_class_boost.predict(X_test)
MAE_class_boost = mean_absolute_error(y_test,y_predict)


a=len(y_predict)
portfolio = [[] for i in range(a)]
for i in range(a):
	index = y_predict[i]
	name = str(names_test[i])
	portfolio[i] += name,
	portfolio[i] += solvers[index],

with open('rf_class_boost_portfolio_%s.csv' %(stamp),'w') as f:
	writer = csv.writer(f)
	writer.writerows(portfolio)

rf_class_boost_reduced = GradientBoostingClassifier(**params)
rf_class_boost_reduced.fit(X_train_reduced,y_train)
y_predict = rf_class_boost_reduced.predict(X_test_reduced)
MAE_class_boost_reduced = mean_absolute_error(y_test,y_predict)

a=len(y_predict)
portfolio = [[] for i in range(a)]
for i in range(a):
	index = y_predict[i]
	name = str(names_test[i])
	portfolio[i] += name,
	portfolio[i] += solvers[index],

with open('rf_class_boost_reduced_portfolio_%s.csv' %(stamp),'w') as f:
	writer = csv.writer(f)
	writer.writerows(portfolio)



###################################################################

# Section 5: Print important stats to screen

###################################################################
print()
print('The Random Forest predicts the dimension of features to train on is around %s' %(n_features_class))
print()
#print('The out-of-bag percent-error for the Full Random Forest is = %.4f' %(error))
#print('The out-of-bag percent-error for the Reduced Random Forest is = %.4f' %(error_reduced))
print()
print('MAE Regression Forest:                         %0.3f, %0.3f' %(MAE_regress,error_regress))
print('MAE Reduced Regression Forest:                 %0.3f, %0.3f' %(MAE_regress_reduced,error_regress_reduced))
print('MAE Boosted Regression Forest:                 %0.3f' %(MAE_regress_boost))
print('MAE Boosted Reduced Regression Forest:         %0.3f' %(MAE_regress_boost_reduced))
print()
print('MAE Classification Forest:                     %0.3f, %0.3f' %(MAE_class,error_class))
print('MAE Reduced Classification Forest:             %0.3f, %0.3f' %(MAE_class_reduced,error_class_reduced))
print('MAE Boosted Classification Forest:             %0.3f' %(MAE_class_boost))
print('MAE Boosted Reduced Classification Forest:     %0.3f' %(MAE_class_boost_reduced))
print()

###################################################################

# Section 6: Make some actual predictions on data without performance metric

###################################################################

# Apply The Full Featured Classifier To The Test Data








'''
method = input("Use full or reduced random forest for predictions on the unlabeled data? ")
print()
solver_number = input("How many top performing solvers do you want to include in the portfolio? ")
print()


features_PREDICT = []
features_PREDICT = X_test


if 'full' in method:
	y_PREDICT = rf.predict(features_PREDICT)
else:
	features_PREDICT = selector.transform(features_PREDICT)
	y_PREDICT = rf_reduced.predict(features_PREDICT)

a=len(y_PREDICT)


portfolio = [[] for i in range(a)]
for i in range(a):
	indices = (y_PREDICT[i]).argsort()[:int(solver_number)]
	name = str(instances_PREDICT[i])
	portfolio[i] += name,
	for index in indices:
		portfolio[i] += solvers[index],

with open('RF_portfolio.csv','w') as f:
	writer = csv.writer(f)
	writer.writerows(portfolio)

'''

###################################################################
###################################################################
###################################################################






























## Poem ## 
# "A planet doesn't explode of itself," said drily
# The Martian astronomer, gazing off into the air --
# "That they were able to do it is proof that highly 
# Intelligent beings must have been living there." 
# --"Earth" by John hall Wheelock