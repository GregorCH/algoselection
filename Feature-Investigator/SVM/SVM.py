##################################################
#
# SVM.py
#
# This script analyzes the GRIPS 2017 Synlab data
# by using recursive SVM and then plotting the
# scalar hyperplane projections with the intention
# of trying to identify clusters/geometry that will be
# helpful in rote classification of the MIP instances
#
# Written: David Haley
# August 2017
#
##################################################


DATA_FILE = "mipdev-features-presol-off-default-fast.csv"
LABELS_FILE = "mipdev_feasible_count_ok.csv"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import preprocessing
from sklearn.svm import SVC

#orthogonal projection of p onto ax + b = 0
def ortho_proj(p,a,b):
	p = np.array(p)
	a = np.array(a)
	return p - a*(np.inner(a,p) + b)/np.inner(a,a)

#read the features into a data frame
features_df = pd.read_csv(DATA_FILE,index_col=0)

#read the labels into a data frame
labels_df = pd.read_csv(LABELS_FILE,index_col=0)

#merge the two data frames together
merged_df = pd.merge(features_df,labels_df,left_index=True, right_index=True)

#the names of the features come from the features file
features = features_df.columns

#the names of the algorithms come from the labels file
labels = labels_df.columns


features_cols = range(0,len(features))
labels_cols = range(len(features), len(features)+len(labels))


X = np.array(merged_df.iloc[:,features_cols])

#using the default sklearn preprocessor does not work well because our feature data
# contains very large numbers in some cases
#X = preprocessing.scale(X)				#scale to std normal

#need to get this to avoid the extreme outliers
sigma = np.array([1]*X.shape[1])        #initialize sigma (default to 1)
for i in range(X.shape[1]):   			#for each column in our data
	col = [xi for xi in X[:,i] if np.abs(xi) < 1e10]
	if len(col) > 0:
		sigma[i] = max(np.std(col),1.0)

#prescale by applying sigmoid to the features
X = np.array([1 / (1 + np.exp(-np.divide(xi,sigma))) for xi in X])

#now do the usual scaling
X = preprocessing.scale(X)

for label in labels:
	y = np.array(merged_df.loc[:,label])  	#labels_cols[label]
	y = [yi > 0 for yi in y]				#classify on any successes whatsoever



	#fit the SVM
	SVM_model = SVC(kernel='linear')
	SVM_model.fit(X,y)

	#what features were used in the hyperplane?  This is what we want to extract

	a = SVM_model.coef_[0]
	b = SVM_model.intercept_[0]

	#find the distances to the first hyperplane
	distances = SVM_model.decision_function(X)

	#now let's project everything onto the first hyperplane
	X = [ortho_proj(p,a,b) for p in X]
	
	#and now re-separate with a (new) SVM
	SVM_model2 = SVC(kernel='linear')
	SVM_model2.fit(X,y)


	#what is the distance to this hyperplane?
	distances2 = SVM_model2.decision_function(X)


	#plot in 2D the distances to the respective hyperplanes
	#colors = ['blue' if yi else 'red' for yi in y]
	#markers = ['o' if yi else 'x' for yi in y]
	for i in range(len(y)):
		if y[i]:
			plt.scatter(distances[i], distances2[i], marker='o', facecolors='None', edgecolors='blue')
		else:
			plt.scatter(distances[i], distances2[i], marker='x', color='red')
	#plt.scatter(distances, distances2, marker=markers, color=colors)
	plt.axis('tight')
	plt.title(label)
	plt.show()


#for testing purposes only
#
#print(features.head())
#print(labels.head())
#print([x for x in features.columns])
#print([x for x in labels.columns])
#print(merged_df.head())
#print(features)
#print(labels)
#print(features_cols)
#print(labels_cols)
#print((merged_df.iloc[:,features_cols]).head())
#print((merged_df.iloc[:,labels_cols]).head())
#print(y)


