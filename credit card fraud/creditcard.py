# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:07:23 2019

@author: KIIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('creditcard.csv')
 
X = dataset.iloc[:,:-1].values
y= dataset.iloc[:, -1].values;

#splitting dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=1/3,random_state=0)

#scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#applying naive bayes algorithm
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#prediction 
y_pred = classifier.predict(X_test)

#checking the accuracy
from sklearn.metrics import  confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Applying Kfold cross validation
from sklearn.model_selection import cross_val_score
accuries = cross_val_score(estimator=classifier,X=X_train,y = y_train,cv =10)

