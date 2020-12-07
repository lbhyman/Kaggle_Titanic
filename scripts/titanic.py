# TODO: pull information from ticket

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

encoder = OneHotEncoder()

def process_data(filename,fit_encoder=True):
    df = pd.read_csv(filename)
    # Not relevant - drop
    df.drop('Ticket',axis=1,inplace=True)
    # Drop 2 instances from embarked
    df.dropna(subset=['Embarked'],inplace=True)
    # Impute Age
    median_age = df['Age'].median()
    df['Age'].fillna(median_age,inplace=True)
    # Impute Fare
    median_fare = df['Fare'].median()
    df['Fare'].fillna(median_fare,inplace=True)
    # Handle categorical attributes
    df['CabinLetter'] =  df['Title'] = ""
    df['CabinLetter'] = list(df['Cabin'].apply(get_cabin_letter))
    df['Title'] = list(df['Name'].apply(get_title))
    X_cat = df[['Pclass','Sex','Embarked','CabinLetter','Title']]
    if fit_encoder:
        encoder.fit(X_cat)
    X_cat = encoder.transform(X_cat).toarray()

    # Handle numerical attributes
    df['CabinNumber'] = ""
    df['CabinNumber'] = list(df['Cabin'].apply(get_cabin_number))
    median_cabin = df['CabinNumber'].median()
    df['CabinNumber'].fillna(median_cabin,inplace=True)
    X_num = df[['Age','SibSp','Parch','Fare','CabinNumber']]
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    # Final X matrix
    X = np.hstack((X_cat,X_num))
    # Final y array
    if 'Survived' in df:
        y = df['Survived'].array
    else:
        y = None
    return X, y

def get_title(input):
    input = str(input)
    title = input.split('.')[0].split(' ')[-1]
    if title == 'Don':
        title = 'Mr'
    if title == 'Dona':
        title = 'Mrs'
    if title == 'Mlle':
        title = 'Miss'
    return title

def get_cabin_letter(input):
    input = str(input)
    if input == 'nan':
        return 'Undefined'
    return input[:1]

def get_cabin_number(input):
    input = str(input)
    if input == 'nan':
        return np.nan
    try:
        return int(input.split(' ')[0][1:])
    except:
        return np.nan

# SVM
'''
# Fit model
model = svm.SVC()
#model = svm.SVC(kernel='rbf',C=1000,gamma=0.01)
# 1000, 0.01

grid = [{'kernel': ['rbf'], 'gamma': [0.075,0.1,0.125],'C':[7,7.5,8,8.5,9]}]
search = GridSearchCV(model,grid,cv=10,scoring='accuracy',return_train_score=True)
search.fit(X,y)
print(search.best_params_)
#model = svm.SVC(kernel='poly',degree=2,C=2.25)
# 0.8290347293156282
model = svm.SVC(kernel='rbf',gamma=0.1,C=8)
# 0.8324438202247191
scores = cross_val_score(model,X,y,cv=10)
print(np.mean(scores))'''

X, y = process_data('../data/train.csv')
print(X[0])
model = svm.SVC(kernel='rbf',gamma=0.1,C=8)
scores = cross_val_score(model,X,y,cv=10)
print(np.mean(scores))

# Random Forest
X, y = process_data('../data/train.csv')
'''
model = RandomForestClassifier()

grid = [{'n_estimators': [100,250,300,350,500], 'max_depth': [2,5,7,8,9,10]}]
search = GridSearchCV(model,grid,cv=10,scoring='accuracy',return_train_score=True)
search.fit(X,y)
print(search.best_params_)'''



model = RandomForestClassifier(n_estimators=300,max_depth=8)
# 0.8335163432073545

scores = cross_val_score(model,X,y,cv=10)
print(np.mean(scores))




model.fit(X,y)

X_test, y_test = process_data('../data/test.csv',fit_encoder=False)

predictions = model.predict(X_test)
indices = list(range(892,1310))
output_df = pd.DataFrame(list(zip(indices,predictions)),columns=['PassengerId','Survived'])
output_df.to_csv('../predictions/RF02.csv',index=False)