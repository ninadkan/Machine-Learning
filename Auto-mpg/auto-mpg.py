# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:48:14 2018

@author: ninadk
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('auto-mpg.data.formatted', delimiter = '\t', header=None)
# print(dataset.describe())

# Taking care of missing data
dataset[3] = dataset[3].replace('?', 'NaN')
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, 0].values

imputer = imputer.fit(X[:,2:3])
X[:,2:3] = imputer.transform(X[:,2:3])


# origin - column (7) had discrete values 1,2,3. That is country of origin. 
# year of manufacture (8) - continuous values. That needs to be scaled
# cylinders has value of 3,4,5,6,8. I think we should just scale that. No need for encoding? 
# We should try couple of approach. For both cases, we'd just be doing one 
# less column than required
#           Column (1), Column (7) Result
# Encoded       N           N
# Encoded       Y           N
# Encoded       N           Y
# Encoded       Y           Y

"""
# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

onemoreHotEncoder = OneHotEncoder(categorical_features = [-1])
X = onemoreHotEncoder.fit_transform(X).toarray()
# Avoiding the Dummy Variable Trap
X = X[:, 1:]
"""

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

all_regressors = ['simple_linear_regression', 
                  'multiple_linear_regression', 
                  'support_vector_regression', 
                  'decision_tree_regression',
                  'random_forest_regression', 
                  'polynomial_regression', ]

len_of_regressors = len(all_regressors)
import linear_regressors as lr

for index in range(0 , len_of_regressors):
    print('Regressor used = ' + all_regressors[index])
    if (index ==0 or index == 1):
        
        print(lr.linear_regressor_processing(X_train, y_train, X_test, y_test))   
        # find out the best parameter fits
        # elapsed time = 0 seconds 
        # Best parameters = {'fit_intercept': True, 'normalize': False}
        # print("Best parameters = %s" % lr.linear_regression_param_selection(X_train, y_train, 10))
        
    elif (index ==2):
        lr.svr_regressor_processing(X_train, y_train, X_test, y_test)
        # find out the best parameter fits
        # elapsed time = 5.5 seconds
        # Best parameters = {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} 
        # print("Best parameters = %s" % lr.svc_param_selection(X_train, y_train_svm, 10))
        
    elif (index ==3):
        lr.decision_tree_regressor_processing(X_train, y_train, X_test, y_test)
    
        # find out the best parameter fits
        # elapsed time = 0.0
        # Best parameters = {'criterion': 'mse', 'max_depth': 5}
        # print("Best parameters = %s" % lr.decision_tree_regression_param_selection(X_train, y_train_svm, 10))
        
    elif (index ==4):
        lr.random_forest_regressor_processing(X_train, y_train, X_test, y_test)
      
        # find out the best parameter fits
        # elapsed time = 5.5 minutes
        # Best parameters = {'criterion': 'mse', 'max_depth': 10, 'n_estimators': 200}
        # print("Best parameters = %s" % lr.random_forest_param_selection(X_train, y_train, 10))
        
    elif (index ==5):
        poly_regresssor = lr.polynomial_regressor() 
        X_poly = poly_regresssor.fit_transform(X_train)
        regressor = lr.svr_regressor()  
        regressor.fit(X_poly, y_train)
        y_pred = regressor.predict(poly_regresssor.fit_transform(X_test))
        # The mean squared error
        print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % regressor.score(poly_regresssor.fit_transform(X_test), y_test))        
    else:
       print(lr.linear_regressor_processing(X_train, y_train, X_test, y_test))        
       # print(lr.linear_regression_param_selection(X_train, y_train, 10))
 

"""    
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Outputting the test set results
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regressor.score(X_test, y_test))
""" 
# Visualising the Training set results
# we don't have two variables, so this does not work correctly
# plt.scatter(X_train, y_train, color = 'red')
"""
X_grid = np.arange(0, len(y_test), 1) # choice of 1 instead of default 0.1 step 
X_grid = X_grid.reshape((len(X_grid), 1))
    
plt.scatter(X_grid, y_test, color = 'blue')
plt.scatter(X_grid, y_pred, color = 'red')

plt.title('Actual (Blue) vs Predicted (Red) results')
plt.xlabel('Total Number of results')
plt.ylabel('Predictions')
plt.show()
"""
# Lets test the results for what we've got

import statsmodels.formula.api as sm
X_out = np.append(arr = np.ones((len(X), 1)).astype(int), values = X, axis = 1)


X_opt = X_out[:, [0, 1, 2, 3, 4, 5, 6,7]]
regressor_OLS = sm.OLS(endog = y.astype(float), exog = X_opt.astype(float)).fit()
print(regressor_OLS.summary())
print(regressor.score)


