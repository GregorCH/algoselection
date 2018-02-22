import _pickle as serializer
import matplotlib.pyplot as plt
import os, string, sys
import numpy as np
import pandas as pd
import math
import csv
import numbers
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib.patches import FancyArrowPatch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
from sklearn.metrics import euclidean_distances
from sklearn import manifold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import sys

from .. import config, logger
from . import utilities
from ..performance import measurement as pm

log = logger._Logger.get_logger(__name__) # set module name for logging
cfg = config.global_config

'''
Train a classifier on a range of different train/test seeds using HYDRA boosting
method.

:param features_file: file (with a path) containing the features
:param performance_file: file (with a path) containing performance
:param method: decision making mechanism for predict:

    - 0 = random forrests  - default one
    - 1 = K neighbors
    - 2 = DecisionTree
    - 3 = SVC  but ut does not work always
    - 4 = linear SVC  but ut does not work always
    - 5 = log regression
:param iterate_up_to_seed: upper limit for random seed that will be used when
    splitting data into train/test sets. Range [0, iterate_up_to_seed) is used.
:param importance_analysis: if important features should be printed out
'''
def train(features_file, performance_file, method = 0, iterate_up_to_seed = 100, importance_analysis = 0):
    for random_seed in range(0, iterate_up_to_seed):
        train_on_single_seed(features_file, performance_file, method, random_seed, importance_analysis)


def train_on_single_seed(  features_file,  performance_file,method = 0 ,random_seed = 0, importance_analysis=0 ):
    '''
    .. warning:: TODO function description.

    :param features_file: file (with a path) containing the features
    :param performance_file: file (with a path) containing performance
    :param method: decision making mechanism for predict:

        - 0 = random forrests  - default one
        - 1 = K neighbors
        - 2 = DecisionTree
        - 3 = SVC  but ut does not work always
        - 4 = linear SVC  but ut does not work always
        - 5 = log regression
    :param random_seed: random seed for the test / train split
    :param importance_analysis: if important features should be printed out
    '''

    NALG =14 # number of algorithms under  consideration
    K= 14 # for saving k best
    seed_number =random_seed
    # file reading
    performance  = pd.read_csv(performance_file)
     #   here could be some problems if there is something inconsistant with the csv files
    features = pd.read_csv(features_file, error_bad_lines=False)
    features = features.rename(columns={'instance_name': 'Problem Name'})

    #  marging to operate on common subset
    total=pd.merge(features, performance,  on='Problem Name')
    features = total.loc[:,features.columns]
    #scaling
    NFEATURES= features.shape[1]-1
    features[features.columns[1:NFEATURES]]= preprocessing.scale(features[features.columns[1:NFEATURES]])
    #preapre data frames
    random_jungle = dict()
    # split
    features, test  = train_test_split(features, test_size = 0.2,random_state = seed_number)
    # voting frame prepare
    optimal= pd.DataFrame()
    optimal['Problem Name']  = test['Problem Name']
    optimal= pd.merge(optimal, performance, on='Problem Name')
    #preparing the data frame with votes
    votes = pd.DataFrame()
    votes= optimal
    votes.ix[:3996, 1:156]= 0
    total=pd.merge(features, performance,  on='Problem Name')
    # jungle creation
    for i in range(1,NALG):
        for j in range(i+1,NALG+1):
            name_j = votes.columns[j]
            name_i = votes.columns[i]
            totaj_ij= pd.concat([total[features.columns],total[name_j],total[name_i]],  axis=1)
            random_jungle = adding_to_the_jungle(name_i,name_j,i,j,totaj_ij, random_jungle,method)
    print('         =======================Jungle created=================== ')
    # voting
    votes =  voting(random_jungle,votes,NALG,test)
    print( '         ======================= Voting  done =================== ')
    # prepare the output
    output= pd.DataFrame()
    votes= votes.drop([votes.columns[0]],axis =1 )

    for i in range (0,test.shape[0]):
        bests = votes.columns[votes.ix[i].argsort()[::-1]]
        output= output.append([bests], ignore_index=True)
    output= pd.concat( [optimal['Problem Name'], output], axis = 1)
    #print(output)
    output = output.set_index(['Problem Name'])

    if method == 1:
        utilities.save_classifier(random_jungle, 'hydra_kneighbors')
        file_name = os.path.join(
                        cfg.models_results_dir,
                        'KNeighborsClassifier_s'+ str(seed_number) + '.csv'
                    )
    elif method == 2:
        utilities.save_classifier(random_jungle, 'hydra_decisiontree')
        file_name = os.path.join(
                        cfg.models_results_dir,
                        'DecisionTreeClassifier_s'+ str(seed_number) + '.csv'
                    )
    elif method == 3:
        utilities.save_classifier(random_jungle, 'hydra_svc')
        file_name = os.path.join(
                        cfg.models_results_dir,
                        'SVC_s'+ str(seed_number) + '.csv'
                    )
    elif method == 4:
        utilities.save_classifier(random_jungle, 'hydra_linearsvc')
        file_name = os.path.join(
                        cfg.models_results_dir,
                        'LinearSVC_s'+ str(seed_number) + '.csv'
                    )
    elif method == 5:
        utilities.save_classifier(random_jungle, 'hydra_logregression')
        file_name = os.path.join(
                        cfg.models_results_dir,
                        'LogisticRegression_s'+ str(seed_number) + '.csv'
                    )
    else:
        utilities.save_classifier(random_jungle, 'hydra_randomforest')
        file_name =  os.path.join(
                        cfg.models_results_dir,
                        'RandomForestClassifier_s'+ str(seed_number) + '.csv'
                     )

    output.to_csv(file_name ,columns=[output.columns[i] for i in range(0,K)],header=False)
    if importance_analysis == 1:
         a = importance_features(random_jungle,NALG,features)
    return output

def predict(instance_name, feature_data, performance_data):
    '''
    .. warning:: TODO function description.
    '''
    # to load trained model use utilities.load_classifier
    # example: classifier = utilities.load_classifier('hydra_randomforest')
    # this will search for file named hydra_randomforest.model in directory
    # pointed by Config's models_dir parameter.
    # By default this is set to gripsPredictorPkg/data/models

    # suggest that feature_data and performance_data are of list type
	# (don't have better proposal right now)

    raise NotImplementedError('This method should use some of the trained models, \
        problem instance features and performance data to predict algorithm ranking')

def adding_to_the_jungle(name_i, name_j,i,j,Xij, jungle,method):
    '''Adding  decision making mechanism to the jungle to the jungle.

    :param name_i: name of the first algorithm in the  pairwise  jungle creatuion
    :param name_j: name of the second algorithm in the  pairwise  jungle creatuion
    :param i: the number of the first used onlu for naming the decission making mechanism .
    :param j: the number of the second used onlu for naming the decission making mechanism .
    :param Xij: data frame containing only i-j data for features and performance
    :param jungle: the dictionary we are adding new mechanism to
    :param method: decision makingmechanism we used

    '''
    # adding a column saing if i( =0) or j( =1) is better
    Xij['better'] = 0 ;   #set everything to 0
    Xij.loc[ Xij[name_j]- Xij[name_i]<= 0, 'better'] = 1   # if time_j < time_i set 1
    # setting weights
    Xij['weights'] = abs(Xij[name_j]-Xij[name_i])
    weights= Xij['weights'].as_matrix(columns=None)

    # estracting data to train on
    x_train = Xij.drop(['Problem Name',name_j,name_i,'better','weights'],axis =1 )
    x_train[np.isnan(x_train)] = 1 # fixing the Nan  if needed
    y_train =  Xij['better']
    # preparing forrests
    forrest_name = 'forrest_' + str(i)+'_'+str(j) # name
   #print(y_train)
    if method == 1:
        forrest_ij= neighbors.KNeighborsClassifier(11,weights='distance')
        forrest_ij.fit(x_train,y_train)
    elif method == 2:
        forrest_ij=DecisionTreeClassifier(criterion='entropy')
        forrest_ij.fit(x_train,y_train)
    elif method == 3:
        forrest_ij=  SVC()
        forrest_ij.fit(x_train,y_train)
    elif method == 4:
        forrest_ij=  LinearSVC()
        forrest_ij.fit(x_train,y_train)
    elif method == 5:
        forrest_ij=  LogisticRegression()
        forrest_ij.fit(x_train,y_train)
    else:
        forrest_ij= RandomForestClassifier(n_estimators=100,max_features='sqrt',n_jobs=1,random_state =0)
        forrest_ij.fit(x_train,y_train,sample_weight=weights) # training

    #adding to the jungle
    jungle[forrest_name]=  forrest_ij
    return jungle

def voting(jungle, votes,NALG,test):
    '''
    .. warning:: TODO function description.

    :param jungle: the dictionary we use to vote
    :param votes: empty data frame we  use to collect the votes
    :param NALG: number of algorithms we consider
    :param test: test data frame used  for prediction
    '''
    test = test.drop(['Problem Name'],axis=1)
    for  i in range(1,NALG):
        for j in range(i+1,NALG+1):
            forrest_name = 'forrest_' + str(i)+'_'+str(j)
            pred=jungle[forrest_name].predict(test)
            votes[votes.columns[j]] = votes[votes.columns[j]] + pred
            votes[votes.columns[i]] = votes[votes.columns[i]] + abs(pred-1)
    return votes

def importance_features(random_jungle, NALG,features):
    '''

    .. warning:: TODO function description.

    :param random_jungle: jungle created in hydra
    :param NALG: number of algorithms
    :param features: full list of features to be analysed
    '''
    IMPORTANCE_FEATURES_FILEPATH = os.path.join(cfg.models_results_dir, 'random_forrest.csv')

    FI = pd.DataFrame();
    FI['Feature Name'] = features.columns.values
    FI =  FI.ix[1:]
    FI['importance']=0
    for  i in range(1,NALG):
        for j in range(i+1,NALG+1):
            forrest_name = 'forrest_' + str(i)+'_'+str(j)
            FI['importance'] = FI['importance']  + random_jungle[forrest_name].feature_importances_
    FI= FI.sort_values('importance',axis=0, ascending=False)
    mean_importance = FI.mean(axis=0)
    important_features = FI.loc[ FI['importance']> 2*mean_importance.ix[0,1] ]
    print(important_features)  # if time_j < time_i set 1
    important_features.to_csv(IMPORTANCE_FEATURES_FILEPATH)
    return FI
