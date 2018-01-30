# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('CarCopy.data.csv')
""" This is column 1 to 6 """
X = dataset.iloc[:, 0:6].values 
""" This is the seventh column """
y = dataset.iloc[:, 6].values 

# Taking care of missing data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
# interesting the labelencoder only does transformation of an array. 

for i in range(0, 6):
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    
#len(X)

#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
    
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


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

# Fitting Decision Tree Classification to the Training set
# Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 15, random_state = 0)
"""
Total Accurate Results = 
338
346
0.976878612717



# AND OUR ANSWER IS
#best_accuracy
#Out[3]: 0.98118668596237335
#best_parameters
#Out[4]: {'criterion': 'entropy', 'max_depth': 15}
"""

# Naive_Bayes provides the accuracy of 61%!!! Ignoring this completely
# and it has no parameters that can be tweaked, interesting 
# naive_bayes classifier. 
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()

# Random Forest classifier
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier( n_estimators = 10, criterion = 'entropy', random_state=0)
# best answer 
"""
0.979015918958 <-- accuracy
{'criterion': 'entropy', 'max_depth': 12, 'max_features': 'auto', 'n_estimators': 100}
"""

# Support Vector Machine classifier 
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state =0)
""" 
Total Accurate Results = 
242
346
0.699421965318
Best parameter answer = It took bloody ages to return so ignore the answer. as it is the accuracy is shot 
"""

# Support Vector Machine classifier 
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
"""
Total Accurate Results = 
227
346
0.656069364162
"""



classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

TotalAccurateResult = 0

TotalSum = 0
for j in range(0, 4):
        for k in range(0,4):
            TotalSum = TotalSum + cm[j,k]
            if (j==k):
                TotalAccurateResult = TotalAccurateResult + cm[j,k]
                
print('Total Accurate Results = ') 
print(TotalAccurateResult)
print(TotalSum)            
print(TotalAccurateResult/TotalSum)  

# Applying K-fold validation
# Not sure what it proves, but lets have a go anyway
# Interesting that the result is higher > 97% and the mean of accuracies = 98.11% which is higher than what 
# you'd get if you do the just standard way of getting things done. 
# accuracies.std()
#Out[2]: 0.013497132737898786
#accuracies.mean()
#Out[3]: 0.98116334461106813
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())

"""
# Applying Grid Search to find the best model and the best parameters
#Disabling this code block as its very specific to the classifier that is choosed
from sklearn.model_selection import GridSearchCV

# decision_tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
# random_forest_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150], 'n_estimators': [10, 100, 200], 'max_features': ['auto', 'sqrt', 'log2']}
# kernel_parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#                      {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

# svm_para = [
#        {'kernel': ['rbf'], 'gamma': [0.1, 0.2], 'C': [1, 10, 100, 1000]},
#        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
#        {'kernel': ['sigmoid'], 'gamma': [0.1, 0.2],'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = svm_para,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_accuracy)
print(best_parameters)

"""

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


"""
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()
"""