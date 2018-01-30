# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:33:10 2018

@author: ninadk
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('iris.data', header=None)
""" This is column 1 to 6 """
X = dataset.iloc[:, 0:4].values 
""" This is the seventh column """
y = dataset.iloc[:, -1].values 


classifier_names = set(y)
print(classifier_names)

# Taking care of missing data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
# interesting the labelencoder only does transformation of an array. 
y = labelencoder_y.fit_transform(y)
    
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""
I should normalize when the scale of a feature is irrelevant or misleading and not normalize 
when the scale is meaningful. In this case the scale refers to the length of the petals and 
are important. So scaling them is not the right way to go. Although as always we'll see the results when 
we scale and not scale the results 
"""
"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
"""
# Applying PCA
# The following  gives me the answer that all of the variables are contributing between 16-17% of significance 
# when I set the value of n_components = None. 
#  Hence it does not make sense to reduce the number of variables!!! Interesting result this. 
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
"""
"""
# Applying LDA
# When setting n_Components = None, I get explained_variance as three values
# 0.837678, 0.162109, 0.00021358. So I guess I can safely ignore the last one and ask for the 
# n_components =2  
# but my accuracy drops to 71% hence, I am not going to use this either. 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
explained_variance = lda.explained_variance_ratio_
"""


all_classifiers = ['logistic_regression_classifier', 
                  'knn_classifier', 
                  'support_vector_classifier', 
                  'kernel_support_vector_classifier',
                  'naive_bayes_classifier', 
                  'decision_tree_classifier', 
                  'random_forest_classifier']

# total classifiers = 7
index = 0
len_of_classifiers = len(all_classifiers)
import nk_classifiers as nk

for index in range(0 , len_of_classifiers):
    print('Classifier used = ' + all_classifiers[index])
    if (index ==0):
        # logistic_regression_classifier
        nk.logistic_regression_processing(X_train, y_train, X_test, y_test, classifier_names)
        print()
    elif (index ==1):
        # knn classifier
        nk.knn_classifier_processing(X_train, y_train, X_test, y_test, classifier_names)
        print()
    elif (index ==2 or index == 3):
        # support_vector_classifier
        nk.support_vector_processing(X_train, y_train, X_test, y_test, classifier_names)
        print()
    elif (index ==4):
        # naive_bayes_classifier
        nk.naive_bayes_processing(X_train, y_train, X_test, y_test, classifier_names)
        print()
    elif (index ==5):
        # decision_tree_classifier
        nk.decision_tree_processing(X_train, y_train, X_test, y_test, classifier_names)
        print()
    elif (index ==6):
        # random_forest_classifier
        nk.random_forest_processing(X_train, y_train, X_test, y_test, classifier_names)
        print()
    else:
        # logistic_regression_classifier
        print()
    index = index +1

nk.plot_results()



"""


# Building the optimal model using Backward Elimination
# P> [t] should be as small as possible. and we should eliminate only those which are 
# closer to 1


import statsmodels.formula.api as sm
X_out = np.append(arr = np.ones((1728, 1)).astype(int), values = X, axis = 1)


X_opt = X_out[:, [0, 1, 2, 3, 4, 5, 6]]
regressor_OLS = sm.OLS(endog = y.astype(float), exog = X_opt.astype(float)).fit()
regressor_OLS.summary()

X_opt = X_out[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y.astype(float), exog = X_opt.astype(float)).fit()
regressor_OLS.summary()

X_opt = X_out[:, [0, 1, 2, 4, 5]]
regressor_OLS = sm.OLS(endog = y.astype(float), exog = X_opt.astype(float)).fit()
regressor_OLS.summary()

# with the following, Adj-RSq decreases from 0.093 --> 0.092. So we need to stop here.
# Following is not going in the right direction, but the previous one is the right answer
X_opt = X_out[:, [0, 1, 2, 4]]
regressor_OLS = sm.OLS(endog = y.astype(float), exog = X_opt.astype(float)).fit()
regressor_OLS.summary()
"""


