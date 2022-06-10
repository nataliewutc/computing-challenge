import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns


#Data cleaning 
data = pd.read_csv('Crystal_structure.csv')
data_replaced = data.copy()
data_without_columns = data_replaced.drop(columns= ['In literature','v(A)','v(B)','τ'])
data_without_rows = data_without_columns.dropna()
data_replace_dash = data_without_rows.replace('-',np.nan)
data_replace_zero = data_replace_dash.replace(0, np.nan)
data_replace = data_replace_zero.dropna()

#add one hot encoder
data_onehotencoded = data_replace.copy()
classes = set(data_onehotencoded['Lowest distortion'])
for cls in classes:
    data_onehotencoded[cls] = data_onehotencoded['Lowest distortion'].apply(lambda x: 1 if x == cls else 0)
del data_onehotencoded['Lowest distortion']
data_onehotencoded.to_csv('Crystal_structure_preprocessed.csv')

#Min-max scaling 
data_minmax = data_onehotencoded.copy()
name_columns = list(data_minmax.columns)
for i in range(3, len(data_minmax.columns)-5):  
    min_value = min(data_minmax[name_columns[i]])
    max_value = max(data_minmax[name_columns[i]])
    diff = int(max_value) - int(min_value)
    data_minmax[name_columns[i]] = data_minmax[name_columns[i]].apply(lambda x: (x - min_value) / diff) 

#Training and testing data split 
y_col = name_columns.pop(13)
X = data_minmax[name_columns].to_numpy()
y = data_minmax[y_col].to_numpy()

training_fraction = 0.1 #can be changed
training_size =int(training_fraction*len(data_minmax))
X_train = X[:training_size]
X_test = X[training_size:]
y_train = y[:training_size]
y_test = y[training_size:]

model = LogisticRegression(multi_class = 'multinomial', solver='saga')
model.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
regressor.score(X_test, y_pred)

