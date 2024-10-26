#!/usr/bin/env python
# coding: utf-8

# #### Suyash Sreekumar
# #### Dataset : Adult Income Dataset
# #### Final Project
# #### Date : 03/18/24 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate, RandomizedSearchCV
from imblearn.over_sampling import SMOTENC
from scipy.stats.mstats import winsorize

from main import df,random_state
from utility.functions import  datapreprocessing,standardize, eval_table, evaluate, table_update, plot_confusion_matrix, classifiers, table_imbalance,table_missing_values,table_outliers,classifiers_weights
np.random.seed(random_state)


# ###  Total rows : 43957

# ================CHALLENGE 1 : Class Imbalance================

# #### To check whether class imbalance is actually a concern or not by adopting different strategies

# ### Workflow for each strategy
# X & y getting from main dataset
# Get Xtrain and Xval using train test split
# Data is standardized
# 
# Strategy 1 
# Xtrain and Xval is subjected preprocessing(Converted categorical to OHE)
# then evaluated
# <br>
# Strategy 2 
# Since data is standardized ,applying SMOTE to it.
# Data is the preprocessed
# Then its evaluated
# <br>
# Strategy 3
# Xtrain and Xval is subjected preprocessing(Converted categorical to OHE)
# Then its evaluated
# Data for each strategy is labelled as X_train_challenge_no_strategy_no
# <br>
# Eg For Challenge 1, strategy1 <br>
# X_train_c1_st1 

# Creating X and y for Challenge 1 

X = df.drop(['income_>50K'], axis =1)
y = df['income_>50K']
X.head()

# Getting list of numerical columns 
int_col = X.select_dtypes(include=['int']).columns

print("Numeric columns in X",int_col)

#Using this data now all the three strategies will be evaluated
#Evaluation approach Train Test Split data 
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=random_state)
X_train,X_val = standardize(X_train,X_val)


# ================Strategy 1 : Do nothing================

# Evaluation of Strategy 1

X_train_c1_st1, X_val_c1_st1 = datapreprocessing(X_train,X_val)
# Evaluate classifiers
results = evaluate(classifiers, X_train_c1_st1, X_val_c1_st1, y_train, y_val,'Class Imbalance','No Change',table_imbalance)

#Printing Evaluation table for model Random Forest
print("Random forest evaluation table")
print(table_imbalance['Random Forest'])

# In[40]:
print("\nChallenge Class Imbalance - Strategy No Change\n",table_imbalance['Random Forest'].iloc[0,4])
# plot_confusion_matrix('Class Imbalance','No Change')

# ================Strategy 2 : Over-Sampling (SMOTE)================

# Applying SMOTENC to the training set
#Getting list of categorical columns t

categorical_columns = X_train.select_dtypes(include=['object']).columns
categorical_indices = [X_train.columns.get_loc(col) for col in categorical_columns]
print(categorical_columns,categorical_indices)
smote = SMOTENC(categorical_features=categorical_indices,random_state=random_state)

#Applying SMOTE on the training dataset
X_train_c1_st2, y_train_c1_st2 = smote.fit_resample(X_train, y_train)
X_train_c1_st2, X_val_c1_st2 = datapreprocessing(X_train_c1_st2,X_val)


print("Missing data count after applying SMOTENC")
print(X_train_c1_st2.isna().sum())

# ================Evaluating SMOTENC Strategy================
# Evaluate classifiers
results_resampled = evaluate(classifiers, X_train_c1_st2, X_val_c1_st2, y_train_c1_st2, y_val,'Class Imbalance','SMOTE',table_imbalance)


# plot_confusion_matrix('Class Imbalance','SMOTE')
print("\nChallenge Class Imbalance - Strategy SMOTE\n",table_imbalance['Random Forest'].iloc[1,4])

# plot_confusion_matrix('Class Imbalance','SMOTE')
table_imbalance['Random Forest']


# ================Strategy 3 Cost Sensitive Learning================


# Preprocessing the data to apply one hot encoding
X_train_c1_st3,X_val_c1_st3=datapreprocessing(X_train,X_val)

#Applying inverse frequency logic

total_samples = len(df)
class_weights = {}

classes = df['income_>50K'].unique()

for cls in classes:
    class_samples = len(df[df['income_>50K'] == cls])
    class_weight = total_samples / (2 * class_samples)  
    class_weights[cls] = class_weight

print("Class Weights:", class_weights)

# ================Evaluating Cost Sensitive Learning Strategy================
results = evaluate(classifiers_weights, X_train_c1_st3, X_val_c1_st3, y_train, y_val,'Class Imbalance','Cost Sensitive Learning',table_imbalance)

print("\nChallenge Class Imbalance - Strategy Cost Sensitive learning\n",table_imbalance['Random Forest'].iloc[2,4])

plot_confusion_matrix('Class Imbalance','Cost Sensitive Learning')

#Printing Final table for Challenge 1 : Class Imbalance
print("Challenge 1 : Class Imbalance")
print(table_imbalance['Random Forest'])


table_imbalance['Dummy Classifier (Stratified)']
