import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import svm
from scipy.stats import norm
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import Union
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from IPython.display import display
from time import time
import matplotlib.widgets
from matplotlib.widgets import RadioButtons, CheckButtons
%matplotlib nbagg 
import matplotlib.animation 
from sklearn.inspection import permutation_importance
import random

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

# Min-max scaling 
def min_max_scaling(data, min_idx=0, max_idx=10):
    data_minmax = data.copy()
    name_columns = list(data_minmax.columns)
    for i in range(min_idx, max_idx):  
        min_value = min(data_minmax[name_columns[i]])
        max_value = max(data_minmax[name_columns[i]])
        diff = int(max_value) - int(min_value)
        data_minmax[name_columns[i]] = data_minmax[name_columns[i]].apply(lambda x: (x - min_value) / diff) 
    return data_minmax

data_minmax = min_max_scaling(ohe_encoded, 0, 10)
    
# Test-train split 
name_columns = list(data_minmax.columns)
X = data_minmax[name_columns[:10]].to_numpy()
y = data_minmax[name_columns[10]].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

# Logistic regression 
logistic = LogisticRegression(max_iter = 1000).fit(X_train, y_train)
logistic_y_pred = logistic.predict(X_test)
logistic_score = logistic.score(X_test, y_test)
print('Logistic Regression Score: %.2f' % (logistic_score*100))

#Random forest 
forest = RandomForestClassifier().fit(X_train,y_train)
forest_y_pred = forest.predict(X_test)
forest_score = forest.score(X_test, y_test)
print('Random Forest Score: %.2f' % (forest_score*100))

#KNN
models = []
k_values = list(range(1,70))
# Testing different k values 
for k in k_values:
    model= KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    models.append(model)
scores = [model.score(X_test, y_test) for model in models]
# Find k value with highest score 
highest_score = max(scores)
index = scores.index(highest_score)
# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=k_values[index]).fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)
knn_score = knn_model.score(X_test, y_test)
print('KNN Score: %.2f' % (knn_score*100))

#SVM 
C = 10.0
# Testing different SVM models 
models = (
    svm.SVC(kernel='linear', C=C),
    svm.LinearSVC(C=C, max_iter=10000),
    svm.SVC(kernel='rbf', gamma=1, C=C),
    svm.SVC(kernel='poly', degree=1.5, gamma='auto', C=C)
)
models = [clf.fit(X_train, y_train) for clf in models]
scores = [clf.score(X_test, y_test) for clf in models]
# Get SVM model with highest score 
highest_score = max(scores)
index = scores.index(highest_score)
# Train SVM model 
svm_y_pred = models[index].predict(X_test)
svm_score = models[index].score(X_test, y_test)
print('SVM Score: %.2f' % (svm_score*100))   

#Classification report 
target_names = ['cubic', 'tetragonal', 'orthorhombic','rhombohedral']
knn_report = metrics.classification_report( y_test, knn_y_pred, target_names=target_names, zero_division = 0) 
forest_report = metrics.classification_report( y_test, forest_y_pred, target_names=target_names, zero_division = 0)
print(f"Random Forest classification report")
print(forest_report)
print(f"KNN classification report")
print(knn_report)

#Cross validation for random forest classifier 

def get_sample_numbers():
    models = dict()
    for s in np.arange(0.1, 1.1, 0.2):
        key = '%.1f' % s
        if s == 1.0:
            s = None
        models[key] = RandomForestClassifier(max_samples=s)
    return models

def get_number_of_features():
    models = dict()
    for i in range(1,6):
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
    depths = [d for d in range(1,6)] + [None]
    for d in depths:
        models[str(d)] = RandomForestClassifier(max_depth=d)
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
        scores = evaluate_model(model, X_test, y_test)
        results.append(scores)
        names.append(name)
        mean.append(np.mean(scores))
        std.append(np.std(scores))
    data = {'Name' : names,
           'Mean' : mean,
           'Standard deviation' : std}
    results = pd.DataFrame(data).rename(columns={"Name": column_name})
    return results 
        
# Get the different hyperparameters to evaluate
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

#Feature importance 
def feature_importance(model, X_test, y_test): 
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    forest_importances = pd.Series(abs(result.importances_mean), index=name_columns[:10])
    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()
    print(f"Most important feature: {forest_importances.idxmax(axis=1)}")
    return 

feature_importance(forest, X_test, y_test)

labels = ["cubic", "tetragonal", "orthorhombic","rhombohedral"]
categories = ('all', 'r(AXII)(Å)', 'r(AVI)(Å)', 'r(BVI)(Å)', 'EN(A)', 'EN(B)', 'l(A-O)(Å)', 'l(B-O)(Å)', 'ΔENR', 'tG', 'μ')

# Dictionary of confusion matrices
confusion_matrix = {}
confusion_matrix[0] = metrics.confusion_matrix(y_test, forest_y_pred, labels=labels)

# Plot confusion matrix 
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.50)

# Get dictionary of confusion matrices, each with a single feature column dropped from the data 
results = categories[slice(1,11)]
for i in results:
    change_data = data_minmax.copy()
    new_data = change_data.drop(columns = i)
    new_names = list(new_data.columns)
    X = new_data[new_names[:9]].to_numpy()
    y = new_data[new_names[9]].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =42)
    # Retrain model with one less feature 
    forest = RandomForestClassifier().fit(X_train, y_train)
    forest_y_pred = forest.predict(X_test)
    index = categories.index(i)
    confusion_matrix[index] = metrics.confusion_matrix(y_test, forest_y_pred, labels=labels)

# Make sure graph of initial full data is displayed 
matrix = confusion_matrix[0]

for y in range(matrix.shape[0]):
    for x in range(matrix.shape[1]):
        ax.text(x, y, matrix[y,x], horizontalalignment='center', verticalalignment='center', color='#8B8000')
        
    # Set radio buttons for the interactive graph 
    radio_ax = plt.axes([0.0, 0.45, 0.3, 0.3])
    radio = RadioButtons(radio_ax, categories)
    
    def callback(label: str='all') -> plt:
        ax.cla()
        index = categories.index(label)
        matrix = confusion_matrix[index]
        ax.imshow(matrix, origin='lower',cmap='hot')
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                ax.text(x, y, matrix[y,x], horizontalalignment='center', verticalalignment='center', color='#8B8000')
        # Set up graph 
        ax.set_xlabel('Real Crystal Structure')
        ax.set_ylabel('Predicted Crystal Structure')
        x_label_list = ['Cubic', 'Tetragonal', 'Orthorhombic', 'Rhombohedral']
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(x_label_list, fontsize=7)
        ax.tick_params(axis="x", labelrotation=-300)
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(x_label_list, fontsize=7)
        plt.show()
        return
    
callback()
radio.on_clicked(callback)





