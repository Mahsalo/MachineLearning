#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:24:13 2020

@author: mahsa
Binary Logistic Regression 
Logistic Regression is a linear classifier and it's good for linearly separable data'
steps:
    1-get the data
    2-split the data
    3-normalize both the test and train data
    4-train the logistic regression model and choose learning rate and an optimizer
    5-evaluate on test, find the score, confusion matrix

"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




#Scikit-learn
#Data
x = np.arange(10).reshape(-1, 1)# columns=features, rows=samples
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

#model
model = LogisticRegression(solver='liblinear', random_state=0, max_iter=100, penalty='l1')
model.fit(x,y)

#model params.
model.intercept_
model.coef_
model.predict_proba(x)#first columns show probability of zero and the second column is prob(1)
pred = model.predict(x)
score = model.score(x,y)#ratio of correct predictions/total observations
cf_mat = confusion_matrix(y,pred) #true on-diag and false off-diag

#Digits Data
x, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test =\
    train_test_split(x, y, test_size=0.2, random_state=0)
scaler = StandardScaler()##standardize the training data
x_train = scaler.fit_transform(x_train)
model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',
                           random_state=0)
model.fit(x_train, y_train)
x_test = scaler.transform(x_test)##standardize the test data
y_pred = model.predict(x_test)
tr_score = model.score(x_train,y_train)
ts_score = model.score(x_test,y_test)
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

