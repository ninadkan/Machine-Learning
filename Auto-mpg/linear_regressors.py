# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:58:04 2018

@author: ninadk
"""
import numpy as np

def simple_linear_regressor():
    # this is same as the multiple linear regressor
    return multiple_linear_regressor()


def multiple_linear_regressor():
    from sklearn.linear_model import LinearRegression
    return LinearRegression()

def polynomial_regressor():
    from sklearn.preprocessing import PolynomialFeatures
    return PolynomialFeatures(degree = 2)

def svr_regressor():
    from sklearn.svm import SVR
    return SVR(kernel = 'rbf', gamma = 0.1, C=10.0)

def decision_tree_regressor():
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(random_state = 0)

def random_forest_regressor():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators = 10, random_state = 0)


def linear_regressor_processing(X_train, y_train, X_test, y_test):
    regressor = multiple_linear_regressor()
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    score = regressor.score(X_test, y_test)
    # Outputting the test set results
    # print('Coefficients: \n', regressor.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % score)
    return score


def svr_regressor_processing(X_train, y_train, X_test, y_test):
    from sklearn.preprocessing import StandardScaler
    sc_y = StandardScaler()
    y_train_svm = sc_y.fit_transform(y_train.reshape(-1,1))
    y_train_svm= y_train_svm.ravel()
    regressor = svr_regressor()  
    regressor.fit(X_train, y_train_svm)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # extra test for the svm
    y_pred = sc_y.inverse_transform(y_pred)
    # The mean squared error
    print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regressor.score(X_test, y_test))
    return
    
    
def decision_tree_regressor_processing(X_train, y_train, X_test, y_test):
    regressor = decision_tree_regressor() 
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # The mean squared error
    print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regressor.score(X_test, y_test))
    return
    
def random_forest_regressor_processing(X_train, y_train, X_test, y_test):
    regressor = random_forest_regressor() 
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # The mean squared error
    print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regressor.score(X_test, y_test))
    return    

def svc_param_selection(X, y, nfolds):
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV

    svc_param = {'C': [0.001, 0.01, 0.1, 1, 10], 
                 'gamma' : [0.001, 0.01, 0.1, 1], 
                 'kernel': ['rbf','sigmoid', 'linear']}
    grid_search = GridSearchCV(estimator = svm.SVR(kernel='rbf'), 
                               param_grid = svc_param, 
                               cv=nfolds)
    grid_search = grid_search.fit(X, y)
    best_accuracy = grid_search.best_score_
    print("Best Accuracy: %.2f" % best_accuracy)
    return grid_search.best_params_

def random_forest_param_selection (X, y, nfolds):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    regressor = RandomForestRegressor(random_state = 0)

    random_forest_para = {'criterion':['mse','mae'],
                          'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150], 
                          'n_estimators': [10, 100, 200]}
    grid_search = GridSearchCV(estimator = regressor,
                               param_grid = random_forest_para,
                               cv = nfolds)
    grid_search = grid_search.fit(X, y)
    best_accuracy = grid_search.best_score_
    print("Best Accuracy: %.2f" % best_accuracy)

    return grid_search.best_params_


def linear_regression_param_selection(X_train, y_train, nfolds):
    from sklearn.model_selection import GridSearchCV
    linear_regressor_para = {'fit_intercept': [True, False], 'normalize': [True, False]}
    grid_search = GridSearchCV(estimator = multiple_linear_regressor(),
                               param_grid = linear_regressor_para,
                               cv = nfolds)
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    print("Best Accuracy: %.2f" % best_accuracy)
    return grid_search.best_params_


def decision_tree_regression_param_selection(X_train, y_train, nfolds):
    from sklearn.model_selection import GridSearchCV
    decision_tree_para = {'criterion':['mse','mae','friedman_mse'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
    grid_search = GridSearchCV(estimator = decision_tree_regressor(),
                               param_grid = decision_tree_para,
                               cv = nfolds,
                               verbose = 4)
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    print("Best Accuracy: %.2f" % best_accuracy)
    return grid_search.best_params_
 
    


    

