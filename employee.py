# -*- coding: utf-8 -*-
"""Employee.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1G67HLQI6_97Oe4NZA6h83m1OCavt_H0G
"""

import pandas as pd

data=pd.read_csv("/content/adult 3.csv")

data

data.shape

data.head()

data.tail(6)

data.isna()

data.isna().sum()

print(data.occupation.value_counts())

print(data.gender.value_counts())

print(data.education.value_counts())

print(data['marital-status'].value_counts())

print(data.workclass.value_counts())

data.occupation.replace({'?':'Others'},inplace=True)

print(data.occupation.value_counts())

data.workclass.replace({'?':'Not listed'},inplace=True)

print(data.workclass.value_counts())

data=data[data['workclass']!='Without-pay']
data=data[data['workclass']!='Never-worked']

print(data.workclass.value_counts())

data.shape

data=data[data['education']!='5th-6th']
data=data[data['education']!='1st-4th']
data=data[data['education']!='Preschool']

print(data.education.value_counts())

data.shape

data.drop(columns=['education'],inplace=True)

data

import matplotlib.pyplot as plt
plt.boxplot(data['age'])
plt.show()

data= data[ (data['age']<=75) & (data['age']>=17) ]

plt.boxplot(data['age'])
plt.show()

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['workclass']=encoder.fit_transform(data['workclass'])
data['marital-status']=encoder.fit_transform(data['marital-status'])
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])
data['race']=encoder.fit_transform(data['race'])
data['gender']=encoder.fit_transform(data['gender'])
data['native-country']=encoder.fit_transform(data['native-country'])
data

x=data.drop(columns=['income'])
y=data['income']

x

y

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
x

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest= train_test_split(x,y, test_size=0.2, random_state=23, stratify=y)

xtrain

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(xtrain, ytrain)
predict=knn.predict(xtest)
predict

from sklearn.metrics import accuracy_score
#accuracy_score(ytest,predict)
print("Accuracy:", accuracy_score(ytest, predict))