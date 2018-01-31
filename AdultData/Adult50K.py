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
dataset = pd.read_csv('adult.data', header = None)
dataset_2 = pd.read_csv('adult.test') # not sure what this one is. So ignoring this. 

# taking care of missing data. Replace all. Note, its not ? its '\b?' that needed replacing!!!
dataset = dataset.replace(' ?', 'NaN')

# ::TBD : How to optimize this
X = dataset[[0,1,2,4,5,6,7,8,9,10,11,12,13]].values

categories_to_be_encoded = [1,4,5,6,7,8,12]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

# interesting the labelencoder only does transformation of an array. 
for i in categories_to_be_encoded:
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    
y = dataset.iloc[:, -1].values 
classifier_names = set(y)
print(classifier_names)
# Encoding the Independent Variable 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Taking care of missing data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])

    
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""
I should normalize when the scale of a feature is irrelevant or misleading and not normalize 
when the scale is meaningful. In this case the scale refers to the length of the petals and 
are important. So scaling them is not the right way to go. Although as always we'll see the results when 
we scale and not scale the results 
So, I believe that the continuous variables should be scaled and discrete quantities should be encoded 
and left as they are. scaling discreate quantites makes no difference anyway
"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
# Applying PCA
# The following  gives me the answer that all of the variables are contributing between 16-17% of significance 
# when I set the value of n_components = None. 
#  Hence it does not make sense to reduce the number of variables!!! Interesting result this. 
# [ 0.15763585  0.1039328   0.0925214   0.08437045  0.08277295  0.07860153
#  0.07328428  0.06819979  0.0657659   0.0602016   0.0524953   0.05056106
#  0.0296571 ]
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
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


index = 0
len_of_classifiers = len(all_classifiers)
import nk_classifiers as nk

for index in range(0 , len_of_classifiers):
    print('Classifier used = ' + all_classifiers[index])
    if (index ==0):
        # logistic_regression_classifier
        # nk.logistic_regression_processing(X_train, y_train, X_test, y_test, classifier_names)
        
        """
        done in 98.013s
        Best estimator found by grid search:
        LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
                  intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                  penalty='l2', random_state=0, solver='newton-cg', tol=0.0001,
                  verbose=0, warm_start=False)
        Predicting category on the test set
        done in 0.001s
        accuracy:   0.818
        classification report:
                     precision    recall  f1-score   support
        
               >50K       0.84      0.94      0.89      4918
              <=50K       0.70      0.45      0.55      1595
        
        avg / total       0.81      0.82      0.80      6513
        
        confusion matrix:
        [[4603  315]
         [ 873  722]]
        """
        print()
    elif (index ==1):
        # knn classifier
        # nk.knn_classifier_processing(X_train, y_train, X_test, y_test, classifier_names)
        print()
    elif (index ==2 or index == 3):
        # support_vector_classifier
        nk.support_vector_processing(X_train, y_train, X_test, y_test, classifier_names)
        """
        done in 8833.935s
        Best estimator found by grid search:
        SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
          max_iter=-1, probability=False, random_state=0, shrinking=True,
          tol=0.001, verbose=False)
        Predicting category on the test set
        done in 1.584s
        accuracy:   0.843
        classification report:
                     precision    recall  f1-score   support
        
               >50K       0.87      0.93      0.90      4918
              <=50K       0.73      0.57      0.64      1595
        
        avg / total       0.84      0.84      0.84      6513
        
        confusion matrix:
        [[4588  330]
         [ 690  905]]
        """
        print()
    elif (index ==4):
        # naive_bayes_classifier
        # nk.naive_bayes_processing(X_train, y_train, X_test, y_test, classifier_names)
        """
        done in 0.299s
        Best estimator found by grid search:
        GaussianNB(priors=None)
        Predicting category on the test set
        done in 0.002s
        accuracy:   0.801
        classification report:
                     precision    recall  f1-score   support
        
               >50K       0.81      0.95      0.88      4918
              <=50K       0.70      0.33      0.45      1595
        
        avg / total       0.79      0.80      0.77      6513
        
        confusion matrix:
        [[4691  227]
         [1070  525]]
        """
        print()
    elif (index ==5):
        # decision_tree_classifier
        # nk.decision_tree_processing(X_train, y_train, X_test, y_test, classifier_names)
        """
        done in 48.404s
        Best estimator found by grid search:
        DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=8,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')
        
        accuracy:   0.852
        classification report:
                     precision    recall  f1-score   support
        
               >50K       0.87      0.95      0.91      4918
              <=50K       0.79      0.55      0.64      1595
        
        avg / total       0.85      0.85      0.84      6513
        
        confusion matrix:
        [[4681  237]
         [ 725  870]]
        """
        print()
    elif (index ==6):
        # random_forest_classifier
        # nk.random_forest_processing(X_train, y_train, X_test, y_test, classifier_names)
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


