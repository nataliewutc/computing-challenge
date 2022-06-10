#Data cleaning 
data = pd.read_csv('Crystal_structure.csv')
data_replaced = data.copy()
data_without_columns = data_replaced.drop(columns= ['In literature','v(A)','v(B)','τ','Compound', 'A', 'B'])
data_without_rows = data_without_columns.dropna()
data_replace_dash = data_without_rows.replace('-',np.nan)
data_replace_zero = data_replace_dash.replace(0, np.nan)
data_replace = data_replace_zero.dropna()

'''#add one hot encoder
data_onehotencoded = data_replace.copy()
classes = set(data_onehotencoded['Lowest distortion'])
for cls in classes:
    data_onehotencoded[cls] = data_onehotencoded['Lowest distortion'].apply(lambda x: 1 if x == cls else 0)
del data_onehotencoded['Lowest distortion']
#print(data_onehotencoded)
data_onehotencoded.to_csv('Crystal_structure_preprocessed.csv')'''

#Min-max scaling 
data_minmax = data_replace.copy()
name_columns = list(data_minmax.columns)
for i in range(3, len(data_minmax.columns)-5):  
    min_value = min(data_minmax[name_columns[i]])
    max_value = max(data_minmax[name_columns[i]])
    diff = int(max_value) - int(min_value)
    data_minmax[name_columns[i]] = data_minmax[name_columns[i]].apply(lambda x: (x - min_value) / diff) 
    
#Training and testing data split 
X = data_minmax[name_columns[:10]].to_numpy()
y = data_minmax[name_columns[10]].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# one-hot encode input variables
onehot_encoder = OneHotEncoder(handle_unknown = 'ignore')
onehot_encoder.fit(X_train)
X_train = onehot_encoder.transform(X_train)
X_test = onehot_encoder.transform(X_test)
# ordinal encode target variable
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
classifier = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))


