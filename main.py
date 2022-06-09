import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from typing import Union
from scipy.stats import norm
import seaborn as sns
from typing import List
import scipy.fft

data = pd.read_csv('Crystal_structure.csv')
data_replaced = data.copy()
data_without_columns = data_replaced.drop(columns= ['In literature','v(A)','v(B)','τ'])
data_without_rows = data_without_columns.dropna()
data_replace_dash = data_without_rows.replace('-',np.nan)
data_replace_zero = data_replace_dash.replace(0, np.nan)
data_replace = data_replace_zero.dropna()
data_replace
#add encoder
data_onehotencoded = data_replace.copy()
classes = set(data_onehotencoded['Lowest distortion'])
for cls in classes:
    data_onehotencoded[cls] = data_onehotencoded['Lowest distortion'].apply(lambda x: 1 if x == cls else 0)
del data_onehotencoded['Lowest distortion']
data_onehotencoded
data_onehotencoded.to_csv('Crystal_structure_preprocessed.csv')