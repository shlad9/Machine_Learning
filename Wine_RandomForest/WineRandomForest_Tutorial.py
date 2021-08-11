#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Wine Machine Learning Practice Model
# Sklearn ML tutorial: https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn


# In[1]:


# import statements

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#random forest model
from sklearn.ensemble import RandomForestRegressor

#cross validation pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#import metrics for evaluation
from sklearn.metrics import mean_squared_error, r2_score

#save sklearn models for future use - similar to pickling in python
from sklearn.externals import joblib


# In[8]:



dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep = ";")


# In[14]:


data


# In[10]:


data.shape


# In[12]:


#summary of statistics for dataset
data.describe()


# In[20]:


#seperating the target feature

y = data.quality
X = data.drop("quality", axis = 1)

#set aside 20% for testing the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123, stratify = y )


# In[33]:


#preprocessing of data for standardization purposes, using a Transformer API in sklearn
#  1. Fit the transformer on the training set (saving the means and standard deviations)
#  2. Apply the transformer to the training set (scaling the training data)
#  3. Apply the transformer to the test set (using the same means and standard deviations)

scaler = preprocessing.StandardScaler().fit(X_train)


# In[34]:


#apply Transformer to training data

X_train_scaled = scaler.transform(X_train)

print(X_train_scaled.mean(axis=0))

print(X_train_scaled.std(axis=0))


# In[35]:


#apply Transformer to test data

X_test_scaled = scaler.transform(X_test)

print(X_test_scaled.mean(axis=0))

print(X_test_scaled.std(axis=0))

#The resulting mean and std are not exactly centered at 0, this is because we are using the stats from the training set here too


# In[36]:


pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators = 100))


# In[39]:


#hyperparameter declaration for tuning
#hyperparameters represent "higher-level", structural information about the data, typically set before running the model

pipeline.get_params()


# In[40]:


hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5, 3, 1]}


# In[42]:


#Cross validation

clf = GridSearchCV(pipeline, hyperparameters, cv = 10)

clf.fit(X_train, y_train)

print(clf.best_params_)


# In[45]:


y_pred = clf.predict(X_test)


# In[47]:


print(r2_score(y_test, y_pred))


# In[48]:


print(mean_squared_error(y_test, y_pred))


# In[49]:


#save model to a .pkl file
joblib.dump(clf, 'rf_regressor.pkl')


# In[ ]:




