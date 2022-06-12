import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.stats import norm
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import Union
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from IPython.display import display
from time import time

#Data cleaning 
data = pd.read_csv('Crystal_structure.csv')
data_to_clean = data.copy()
data_without_columns = data_to_clean.drop(columns= ['In literature','v(A)','v(B)','τ', 'Compound'])
data_without_rows = data_without_columns.dropna()
data_replace_dash = data_without_rows.replace('-',np.nan)
data_replace_zero = data_replace_dash.replace(0, np.nan)
data_replace = data_replace_zero.dropna()

#Encoding 
class Encoder:
    def __init__(self, kind: str = 'onehot'):
        # make sure kind is either onehot or label
        assert kind in ['onehot', 'label']
        self.kind = kind
        
    def encode(self, data: pd.Series) -> Union[pd.DataFrame, pd.Series]:
        if self.kind == 'onehot':
            categories = set(data)
            new = pd.DataFrame()
            for column in list(categories):
                new[column] = data.apply(lambda x: 1 if x == column else 0)
        else:
            categories = list(set(data))
            new = data.apply(lambda x: categories.index(x))
            new = pd.DataFrame(new)
            del new[column]
        return new

ohe_encoded = data_replace.copy()
a = Encoder('onehot').encode(ohe_encoded['A'])
b = Encoder('onehot').encode(ohe_encoded['B'])
del ohe_encoded['A']
del ohe_encoded['B']
ohe_encoded = pd.concat([ohe_encoded, a], axis=1)
ohe_encoded = pd.concat([ohe_encoded, b], axis=1)

#Min-max scaling 
data_minmax = ohe_encoded.copy()
name_columns = list(data_minmax.columns)
for i in range(len(data_minmax[:10])):  
    min_value = min(data_minmax[name_columns[i]])
    max_value = max(data_minmax[name_columns[i]])
    diff = int(max_value) - int(min_value)
    data_minmax[name_columns[i]] = data_minmax[name_columns[i]].apply(lambda x: (x - min_value) / diff) 
    
#Training and testing data split 
X = data_minmax[name_columns[:10]].to_numpy()
y = data_minmax[name_columns[10]].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 100)

# Logistic regression 
logistic = LogisticRegression().fit(X_train, y_train)
logistic_y_pred = logistic.predict(X_test)
logistic_score = logistic.score(X_test, y_test)
print('Logistic Regression Score: %.2f' %(logistic_score*100))
logistic_accuracy = accuracy_score(y_test, logistic_y_pred)
print('Logistic Regression Accuracy: %.2f' % (logistic_accuracy*100))

#Random forest 
forest = RandomForestClassifier().fit(X_train,y_train)
forest_y_pred = forest.predict(X_test)
forest_score = forest.score(X_test, y_test)
print('Forest score: %.2f' % (forest_score*100))
forest_accuracy = accuracy_score(y_test, forest_y_pred)
print('Forest Accuracy: %.2f' % (forest_accuracy*100))

#KNN
models = []
k_values = list(range(1,15))
for k in k_values:
    model= KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    models.append(model)
scores = [model.score(X_test, y_test) for model in models]
highest_score = max(scores)
index = scores.index(highest_score)
best_model = KNeighborsClassifier(n_neighbors=k_values[index]).fit(X_train, y_train)
fig, ax =plt.subplots()
ax.bar(k_values, scores)
ax.set_xlabel('Number of Neighbours')
ax.set_ylabel('Score')
knn_y_pred = best_model.predict(X_test)
knn_score = best_model.score(X_test, y_test)
print('KNN Score: %.2f' % (knn_score*100))
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print('KNN Accuracy: %.2f' % (knn_accuracy*100))

#SVM 
C = 5.0
models = (
    svm.SVC(kernel='linear', C=C),
    svm.LinearSVC(C=C, max_iter=10000),
    svm.SVC(kernel='rbf', gamma=0.1, C=C),
    svm.SVC(kernel='poly', degree=1.5, gamma='auto', C=C)
)
models = [clf.fit(X_train, y_train) for clf in models]
scores = [clf.score(X_test, y_test) for clf in models]
highest_score = max(scores)
index = scores.index(highest_score)
svm_y_pred = models[index].predict(X_test)
svm_score = models[index].score(X_test, y_test)
print('SVM score: %.2f' % (svm_score*100))
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print('SVM Accuracy: %.2f' % (svm_accuracy*100))

#Classification report 
target_names = ['cubic', 'tetragonal', 'orthorhombic','rhombohedral']
knn_report = sklearn.metrics.classification_report( y_test, knn_y_pred, target_names=target_names, zero_division = 0) 
forest_report = sklearn.metrics.classification_report( y_test, forest_y_pred, target_names=target_names, zero_division = 0)  
print(knn_report)
print(forest_report)

#Cross validation for random forest classifier 

def get_sample_numbers():
    models = dict()
    # explore ratios from 10% to 100% in 10% increments
    for i in np.arange(0.1, 1.1, 0.2):
        key = '%.1f' % i
        if i == 1.0:
            i = None
        models[key] = RandomForestClassifier(max_samples=i)
    return models

def get_number_of_features():
    models = dict()
    # number of features from 1 to 7
    for i in range(1,8):
        models[str(i)] = RandomForestClassifier(max_features=i)
    return models

def get_number_of_trees():
    models = dict()
    n_trees = [10, 50, 100, 200]
    for n in n_trees:
        models[str(n)] = RandomForestClassifier(n_estimators=n)
    return models

def get_tree_depth():
    models = dict()
    # consider tree depths from 1 to 5 and None=full
    depths = [i for i in range(1,6)] + [None]
    for n in depths:
        models[str(n)] = RandomForestClassifier(max_depth=n)
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X_test, y_test):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

# evaluate the models and store results
def get_scores(models, column_name):
    results, names, mean, std = list(), list(), list(), list()
    for name, model in models.items():
        #start_time = time()
        scores = evaluate_model(model, X_test, y_test)
        #print(name + ' took ' + str(time()-start_time))
        results.append(scores)
        names.append(name)
        mean.append(np.mean(scores))
        std.append(np.std(scores))
    data = {'Name' : names,
           'Mean' : mean,
           'Standard deviation' : std}
    results = pd.DataFrame(data).rename(columns={"Name": column_name})
    return results 
        
# get the different hyperparameters to evaluate
sample_sizes = get_sample_numbers()
sample_size_data = get_scores(sample_sizes, 'Sample sizes')
display(sample_size_data)

number_of_features = get_number_of_features()
num_of_features_data = get_scores(number_of_features, 'Number of features')
display(num_of_features_data)

number_of_trees = get_number_of_trees()
num_of_trees_data = get_scores(number_of_trees, 'Number of trees')
display(num_of_trees_data)

tree_depth = get_tree_depth()
tree_depth_data = get_scores(tree_depth, 'Tree depth')
display(tree_depth_data)






