#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:35:43 2022

@author: Jérémy Peres
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

animalsData = pd.read_csv("data.csv", encoding = 'latin1', engine ='python')
typesData = pd.read_csv("types.csv", encoding = 'latin1', engine ='python')
# animalsData.head()

#Unique types
animalsData.iloc[:,-1].unique()

#Drop column with names
animalsDataNoName = animalsData.drop(animalsData.columns[[0]], axis=1)

plt.figure(figsize=(15,11))
#To avoid heatmap duplication
matrix = np.triu(animalsDataNoName.corr())
sns.heatmap(animalsDataNoName.corr(), annot=True, fmt='.1g', cbar=False, linewidths=0.5, mask=matrix);

#prep data
types = typesData['name'].tolist()
#Attributes of all the animals
X = animalsData.iloc[:,1:-1]
#Types of all the animals
Y = animalsData.iloc[:,-1]

#split + tests
trainX, testX, trainY, testY = train_test_split(X, Y)

#ML test knn
knn = KNeighborsClassifier()
knn.fit(trainX, trainY)


#split + tests
trainX, testX, trainY, testY = train_test_split(X, Y)
#ML test decision tree
tree = DecisionTreeClassifier()
tree.fit(trainX, trainY)
plt.figure(figsize=(15,11))
treePlot = plot_tree(tree, feature_names=X.columns, class_names=types, filled=True)


#split + tests
trainX, testX, trainY, testY = train_test_split(X, Y)
#ML test random forest
rfc = RandomForestClassifier()
rfc.fit(trainX, trainY)


dataHistogram=[['Train','K Neighbors', knn.score(trainX, trainY)*100],
               ['Test','K Neighbors', knn.score(testX, testY)*100],
               ['Train','Decision Tree', tree.score(trainX, trainY)*100],
                ['Test','Decision Tree', tree.score(testX, testY)*100],
                ['Train','Random Forest', rfc.score(trainX, trainY)*100],
                ['Test','Random Forest', rfc.score(testX, testY)*100]]

df=pd.DataFrame(dataHistogram,columns=['Group','Model', 'Score %'])

fig = px.histogram(df, x='Model', y='Score %', color='Group', title="Score comparison between models", barmode='group')
fig.write_html("histogram_models.html")
