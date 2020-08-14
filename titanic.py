# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:46:01 2020

@author: ANKUR
"""
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data.Survived
X = train_data.drop(['Survived'], axis = 1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.6, test_size = 0.4, random_state = 0)

categorical_col = [col for col in X_train.columns if X_train[col].nunique()< 10 and X_train[col].dtype == 'object']
numerical_col = [col for col in X_train.columns if X_train[col].dtype in ['float64', 'int64']]
my_cols = categorical_col + numerical_col
X_train = X_train[my_cols].copy()
X_valid = X_valid[my_cols].copy()
X_test = test_data[my_cols].copy()

numerical_transformer = SimpleImputer(strategy = 'constant')
categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')),('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

performer = ColumnTransformer(transformers = [('num', numerical_transformer, numerical_col), ('cat', categorical_transformer, categorical_col)])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

my_pipeline = Pipeline(steps = [('performer', performer), ('model', model)])
#scores = -1* cross_val_score(my_pipeline, X_train, y_train, cv = 10, scoring = 'neg_mean_absolute_error')
#print(scores.mean())
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)
print("MAE",mean_absolute_error(y_valid, preds))

p = my_pipeline.predict(X_test)
output = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': p})

output.to_csv('submission.csv', index = False)

print("successful")