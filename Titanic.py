# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:35:38 2019

@author: Anish
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("train.csv")
data=data.iloc[:,[1,2,4,5,6,7,9,11]]
data.info()

test_data=pd.read_csv('test.csv')
Z=test_data.iloc[:,0]
test_data=test_data.iloc[:,[1,3,4,5,6,8,10]]
test_data.info()

#Survivors
sns.set_style('whitegrid')
sns.countplot(x='Survived', data= data, palette='RdBu_r')

sns.countplot(x='Survived',hue='Sex',data=data,palette='RdBu_r')
sns.countplot(x='Survived',hue='Pclass',data=data,palette='rainbow')
sns.distplot(data['Age'].dropna(),color='darkred',bins=30)

plt.figure(figsize=[12,10])
sns.boxplot(x='Pclass',y='Age',data=data,palette='rainbow')

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

data['Age'] = data[['Age', 'Pclass']].apply(impute_age, axis = 1)
test_data['Age'] = test_data[['Age', 'Pclass']].apply(impute_age, axis = 1)
data.isnull().sum()
test_data.isnull().sum()

def impute_Pclass(cols):
    Pclass=cols[0]
    if Pclass==3:
        return 1
    elif Pclass==2:
        return 2
    else:
        return 3
data['Pclass']=data[['Pclass']].apply(impute_Pclass, axis=1)
test_data['Pclass']=test_data[['Pclass']].apply(impute_Pclass, axis=1)

sns.distplot(test_data['Fare'].dropna(),color='darkred',bins=30)

def impute_fare(cols):
    Fare=cols[0]
    if pd.isnull(Fare):
        return 10
    return Fare
test_data['Fare']=test_data[['Fare']].apply(impute_fare, axis=1)


X_train=data.iloc[:,1:6].values
y_train=data.iloc[:,0].values
X_test=test_data.iloc[:,0:5].values
#NO y_test
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
X_train[:, 1]=label.fit_transform(X_train[:, 1])
X_test[:, 1]=label.fit_transform(X_test[:, 1])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

df1 = pd.DataFrame(data=Z)
df2 = pd.DataFrame(data=y_pred)
dfs = pd.concat([df1, df2], axis=1)

import tkinter as tk
from tkinter import filedialog
from pandas import DataFrame



dfs = DataFrame(dfs)


root= tk.Tk()

canvas1 = tk.Canvas(root, width =300, height = 300, bg = 'lightsteelblue2', relief = 'raised')
canvas1.pack()

def exportCSV ():
    global dfs
    
    export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
    dfs.to_csv (export_file_path, index = None, header=True)

saveAsButton_CSV = tk.Button(text='Export CSV', command=exportCSV, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=saveAsButton_CSV)

root.mainloop()



















