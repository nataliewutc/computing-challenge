import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from typing import Union
from scipy.stats import norm
import seaborn as sns
from typing import List
import scipy.fft
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Logistic regression 
classifier = LogisticRegression().fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score = classifier.score(X_test, y_test)
print(score)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))

#KNN 

#Random forest 

#SVM 





