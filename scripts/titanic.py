import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def process_data(filename):
    df = pd.read_csv(filename)
    # High NAs - drop cabin
    df.drop('Cabin',axis=1,inplace=True)
    # Not relevant - drop
    df.drop('PassengerId',axis=1,inplace=True)
    df.drop('Name',axis=1,inplace=True)
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
    X_cat = df[['Pclass','Sex','Embarked']]
    encoder = OneHotEncoder()
    X_cat = encoder.fit_transform(X_cat).toarray()

    # Handle numerical attributes
    X_num = df[['Age','SibSp','Parch','Fare']]
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    # Final X matrix
    X = np.hstack((X_cat,X_num))
    print(df.info())
    # Final y array
    if 'Survived' in df:
        y = df['Survived'].array
    else:
        y = None
    return X, y

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

# Random Forest
'''
model = RandomForestClassifier()

grid = [{'n_estimators': [250,300,350], 'max_depth': [7,8,9]}]
search = GridSearchCV(model,grid,cv=10,scoring='accuracy',return_train_score=True)
search.fit(X,y)
print(search.best_params_)'''

X, y = process_data('../data/train.csv')

model = RandomForestClassifier(n_estimators=250,max_depth=9)
# 0.8335163432073545
'''
scores = cross_val_score(model,X,y,cv=10)
print(np.mean(scores))'''

# TODO: pull information from cabin, ticket, and name
model.fit(X,y)

X_test, y_test = process_data('../data/test.csv')

predictions = model.predict(X_test)
indices = list(range(892,1310))
output_df = pd.DataFrame(list(zip(indices,predictions)),columns=['PassengerId','Survived'])
output_df.to_csv('../predictions/RF01.csv',index=False)