# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:00:28 2018

@author: ninadk
"""

def get_logistic_regression_classifier():
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    return classifier
    
def get_knn_classifier():
    # NOTE: Remember to do the feature scaling for this bits to work correctly. 
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski', p=2)
    return classifier

def get_support_vector_classifier():
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state =0)
    return classifier
    
def get_kernel_support_vector_classifier():
    return

def get_naive_bayes_classifier():
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    return classifier
    
def get_decision_tree_classifier():
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
    return classifier

def get_random_forest_classifier():
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier( n_estimators = 10, criterion = 'entropy', random_state=0)    
    return classifier


def logistic_regression_processing(X_train, y_train, X_test, y_test, classifier_names, nfolds = 10):
    logistic_regression_param = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                                 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                 'max_iter' : [100, 200]}

    classifier_processing(X_train, y_train, X_test, y_test, classifier_names, 
                          estimator = get_logistic_regression_classifier(), regression_param = logistic_regression_param,  nfolds= nfolds)    
    return

def knn_classifier_processing(X_train, y_train, X_test, y_test, classifier_names, nfolds = 10):
    
    """
    # Feature Scaling, especially required for the knn classifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    """
    knn_regression_param = {'n_neighbors': [2,5,10], 
                             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                             'p' : [1, 2]}
    
    
    classifier_processing(X_train, y_train, X_test, y_test, classifier_names, 
                          estimator = get_knn_classifier(), regression_param = knn_regression_param,  nfolds= nfolds)
    
    return

def support_vector_processing(X_train, y_train, X_test, y_test, classifier_names, nfolds = 10):
    svm_para = {'kernel': ['rbf', 'linear','sigmoid', 'poly' ], 
                'gamma': [0.1, 0.2, 0.3], 
                'C': [1, 10, 100, 1000]}
    
    classifier_processing(X_train, y_train, X_test, y_test, classifier_names, 
                      estimator = get_support_vector_classifier(), regression_param = svm_para,  nfolds= nfolds)
    return

def kernel_support_vector_processing(X_train, y_train, X_test, y_test):
    # the only difference between kernel and linear vector processing is tht the kernel specified in one is rbf and in other its rbf
    # we don't really care that. Our algorithm finds the best match and uses that to compute the accuracy
    support_vector_processing(X_train, y_train, X_test, y_test)
    return

def naive_bayes_processing(X_train, y_train, X_test, y_test, classifier_names, nfolds = 10):
    # There are no parameter to be passed for this naive bayes - gaussian algorithm 
    param = {}
    classifier_processing(X_train, y_train, X_test, y_test, classifier_names, 
                      estimator = get_naive_bayes_classifier(), regression_param = param,  nfolds= nfolds)
    return

def decision_tree_processing(X_train, y_train, X_test, y_test, classifier_names, nfolds = 10):
    decision_tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
    classifier_processing(X_train, y_train, X_test, y_test, classifier_names, 
                      estimator = get_decision_tree_classifier(), regression_param = decision_tree_para,  nfolds= nfolds)
    return

def random_forest_processing(X_train, y_train, X_test, y_test, classifier_names, nfolds = 10):
    random_forest_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150], 'n_estimators': [10, 100, 200], 'max_features': ['auto', 'sqrt', 'log2']}
    classifier_processing(X_train, y_train, X_test, y_test, classifier_names, 
                  estimator = get_random_forest_classifier(), regression_param = random_forest_para,  nfolds= nfolds)
    return

results = []

def classifier_processing(X_train, y_train, X_test, y_test, classifier_names, estimator, regression_param, nfolds):
    results.append(classifier_processing_wrapper(X_train, y_train, X_test, y_test, classifier_names, estimator, regression_param, nfolds))

def classifier_processing_wrapper(X_train, y_train, X_test, y_test, classifier_names, estimator, regression_param, nfolds):
   
    from time import time
    print("Fitting the classifier to the training set")
    t0 = time()
    from sklearn.model_selection import GridSearchCV
    
    clf = GridSearchCV(estimator = estimator,
                           param_grid = regression_param,
                           scoring = 'accuracy',
                           cv = nfolds)
    
    clf = clf.fit(X_train, y_train)
    train_time = time() - t0
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    
    # #############################################################################
    # Quantitative evaluation of the model quality on the test set
    
    print("Predicting Iris category on the test set")
    t0 = time()
    y_pred = clf.predict(X_test)
    test_time = time() - t0
    print("done in %0.3fs" % (time() - t0))
    
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test, y_pred)
    print("accuracy:   %0.3f" % score)
    print("classification report:")
    print(classification_report(y_test, y_pred, target_names=classifier_names))
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=range(len(classifier_names))))
    
    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time    


def plot_results():
     # make some plots
    import numpy as np
    import matplotlib.pyplot as plt
    indices = np.arange(len(results))
    # UnboundLocalError: local variable 'results' referenced before assignment
    results_local = [[x[i] for x in results] for i in range(4)]
    
    clf_names, score, training_time, test_time = results_local
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)
    
    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    
    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)
    
    plt.show()   
    return 


    



