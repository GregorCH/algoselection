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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# ex. of how to run code:
# python the_comparator.py RF_portfolio.txt performance.csv
# each row in performance.csv is an instance, the columns are performance values for various solvers
# questions can be sent to Alex Georges: ageorges@ucsd.edu

###################################################################

# Section 1A: Grabs Data

###################################################################


PORTFOLIO_PREDICTED=sys.argv[1]
PORTFOLIO_ACTUAL=sys.argv[2]

with open("%s" %(PORTFOLIO_PREDICTED)) as f:
	reader = csv.reader(f, delimiter=",")
	data = list(reader)
portfolio_predicted = [line for line in data] #does include instance name as 0th column
instance_names_predicted = [line[0] for line in data]

solver_number = len(portfolio_predicted[0])-1

with open("%s" %(PORTFOLIO_ACTUAL)) as f:
	reader = csv.reader(f, delimiter=",")
	data = list(reader)
solvers = [line for line in data[0][1:]] #does include instance name as 0th column
instance_names_actual = [line[0] for line in data[1:]]
performance_actual = [line[1:] for line in data[1:]]
performance_actual=[[float(j) for j in i] for i in performance_actual]
performance_actual = np.array(performance_actual)


a=len(instance_names_actual)
portfolio_actual = [[] for i in range(a)]
for i in range(a):
	indices = (performance_actual[i]).argsort()[:int(solver_number)]
	name = str(instance_names_actual[i])
	portfolio_actual[i] += name,
	for index in indices:
		portfolio_actual[i] += solvers[index],

matched_names=[]
portfolio_predicted_matched=[] #not necessary.  just including this for symmetry.
portfolio_actual_matched=[]

for line in portfolio_predicted:
	a=len(line)
	for LINE in portfolio_actual:
		if line[0]==LINE[0]:
			#print('actual: %s' %(LINE))
			#print('predicted: %s'%(line))
			matched_names.append(line[0])
			portfolio_predicted_matched.append(line)
			portfolio_actual_matched.append(LINE)

indices_predicted=[]
indices_actual=[]

for line in portfolio_predicted_matched:
	for solver in line[1:]:
		indices_predicted.append(data[0].index(solver))
for line in portfolio_actual_matched:
	for solver in line[1:]:
		indices_actual.append(data[0].index(solver))

values_predicted=[]
values_actual=[]

for name in matched_names:
	for line in data:
		if line[0]==name:
			for index in indices_predicted:
				values_predicted.append(line[index])
			for index in indices_actual:
				values_actual.append(line[index])

values_diff = [abs(float(i)-float(j)) for i, j in zip(values_predicted,values_actual)]
print(values_diff)