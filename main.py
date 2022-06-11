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










