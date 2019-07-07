# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import make_pipeline
from imblearn.pipeline  import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE

# data preprocessing
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[ : , :-1].values
y= dataset.iloc[:,-1].values
#Visulaizinig the dataset 
fig, ax = plt.subplots(1,1)
ax.pie(dataset.Class.value_counts(),explode=(0,0.1), autopct='%1.1f%%', labels = ['Genuine', 'Fraud'], colors=['y','r'])
plt.axis = 'equal'

#seperating the data set in two parts 
data_frame_1 = dataset[dataset['Class'] == 0]
data_frame_2 = dataset[dataset['Class'] == 1]

#Visualizing frequency distribution 
for i in range(1,29):
    sns.distplot(data_frame_1.iloc[:,i])
    sns.distplot(data_frame_2.iloc[:,i], color='r')
    plt.show()

'''
It can be observed that, frequency distribution for both the classes 
("0" and "1") against features V8, V13, V15, V20, V21, V22, V23, V24,
V25, V26, V27 and V28 are approximately similar. These features would
most certainly not help out for the purpose of differentiating between
class 0 and 1.So, we drop these features from our data_frames so as
to make our model less complex.
'''
data_frame_1 = data_frame_1.drop(columns=["V8","V13","V15","V20","V21","V22","V23","V24","V25","V26","V27","V28"])
data_frame_2 = data_frame_2.drop(columns=["V8","V13","V15","V20","V21","V22","V23","V24","V25","V26","V27","V28"])


for_count_0 = data_frame_1.head(2500)
for_count_1 = data_frame_2.head(50)

count_0_downward = data_frame_1.tail(len(data_frame_1)-2500)
count_1_downward = data_frame_2.tail(len(data_frame_2)-50)



X_train = pd.concat([count_0_downward,count_1_downward])
X_train_set = X_train.drop(columns=['Class']).values
y_train = X_train['Class'].values

X_test = pd.concat([for_count_0,for_count_1])
X_test_set = X_test.drop(columns=['Class']).values
y_test = X_test['Class'].values

#Synthetic Minority Oversampling Technique Algorithm
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

np.count_nonzero(y_res == 0)


#scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

    
#applying naive bayes algorithm
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_res,y_res)

#prediction 
y_pred = classifier.predict(X_test)


#checking the accuracy
from sklearn.metrics import  confusion_matrix
cm = confusion_matrix(y_test,y_pred)


#Applying Kfold cross validation
from sklearn.model_selection import cross_val_score
accuries = cross_val_score(estimator=classifier,X=X_train,y = y_train,cv =10)


