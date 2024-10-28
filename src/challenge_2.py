#!/usr/bin/env python
# coding: utf-8

# #### Suyash Sreekumar
# #### Dataset : Adult Income Dataset
# #### Final Project
# #### Date : 03/18/24 

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from library.data_preprocessing import datapreprocessing, standardize
from library.evaluation import evaluate
from main import df, random_state, classifiers, table_missing_values
np.random.seed(random_state)

'''
Challenge 2 : Missing Values

Standardize the data
Apply train test split

Strategy 1 No Change
apply preprocessing on train;test and train;val
evaluate

Strategy 2 Drop columns
drop missing data rows
apply preprocessing
evaluate

Strategy 3 Impute Data
impute data

apply preprocessing
evaluate
'''
# ================CHALLENGE 2 : Missing Values================

# Getting X and y

X = df.drop(['income_>50K'], axis =1)
y = df['income_>50K']

#Using this data now all the three strategies will be evaluated
#Evaluation approach Train Test Split data 
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=random_state)
X_train,X_val = standardize(X_train,X_val)


# ================Strategy 1 : Do nothing================
X_train_c2_st1,X_val_c2_st1=datapreprocessing(X_train,X_val)

results_missing_values_st1 = evaluate(classifiers, X_train_c2_st1, X_val_c2_st1, y_train, y_val,'Missing Values','No Change',table_missing_values)
print("Missing Values evaluation Table")
print(table_missing_values['Random Forest'])



# ================Strategy 2 : Drop data================

#Need to merge X_train and Y_train so that rows can be dropped 
def merge_df(X_t,y_t):
    # Reset the index of X_train and y_train before merging
    X_t = X_t.reset_index(drop=True)
    y_t = y_t.reset_index(drop=True)

    data_merged=pd.concat([X_t,y_t],axis=1)
    return data_merged

# Combine X and y for each set
train_c2 = merge_df(X_train, y_train)
val_c2 = merge_df(X_val, y_val)

# Drop duplicates
train_c2 = train_c2.dropna()
val_c2 = val_c2.dropna()

# Separate X and y again
X_train_c2_st2 = train_c2.drop(columns=['income_>50K'])
y_train_c2_st2 = train_c2['income_>50K']

X_val_c2_st2 = val_c2.drop(columns=['income_>50K'])
y_val_c2_st2 = val_c2['income_>50K']
del train_c2,val_c2
y_val_c2_st2.isna().sum()

# Data for Deleting missing row
# _,X_testc2_st2=datapreprocessing(X_train_c2_st2,X_test_missing)
# print("--")
X_train_c2_st2,X_val_c2_st2=datapreprocessing(X_train_c2_st2,X_val_c2_st2)


results_missing_values_st2 = evaluate(classifiers, X_train_c2_st2, X_val_c2_st2, y_train_c2_st2, y_val_c2_st2,'Missing Values','Dropping Missing Data',table_missing_values)

print("Missing Values evaluation Table : Drop missing data")
print(table_missing_values['Random Forest'])

# ================Strategy 3 : Imputing Data================

# Creating X and y for  strategy 3
X_train_c2_st3=X_train.copy()
y_train_c2_st3=y_train.copy()
X_val_c2_st3 =X_val.copy()
y_val_c2_st3 = y_val.copy()


imputer = SimpleImputer(strategy='most_frequent')
categorical_columns = ['workclass', 'marital-status', 'occupation','relationship', 'race', 'gender', 'native-country']
print(categorical_columns)

# Fit the imputer on your training data and transform both training and validation data for categorical columns
X_train_c2_imputed_categorical = pd.DataFrame(imputer.fit_transform(X_train_c2_st3[categorical_columns]), columns=categorical_columns)
X_val_c2_imputed_categorical = pd.DataFrame(imputer.transform(X_val_c2_st3[categorical_columns]), columns=categorical_columns)

# Reset the index to default integer index
X_train_c2_st3.reset_index(drop=True, inplace=True)
X_val_c2_st3.reset_index(drop=True, inplace=True)

# Concatenate the imputed categorical columns with the non-categorical columns
X_train_c2_st3= merge_df(X_train_c2_st3.drop(columns=categorical_columns), X_train_c2_imputed_categorical)
X_val_c2_s3 = merge_df(X_val_c2_st3.drop(columns=categorical_columns), X_val_c2_imputed_categorical)

# Data for Deleting missing row
#_,X_testc2_st3=datapreprocessing(X_train_c2_imputed,X_test_missing)
print("--")
X_train_c2_st3,X_val_c2_st3=datapreprocessing(X_train_c2_st3,X_val_c2_st3)

print("All null values now imputed ")
print(X_train_c2_st3.info())

#Evaluating strategy 3
results_missing_values_st2 = evaluate(classifiers, X_train_c2_st3, X_val_c2_st3, y_train_c2_st3, y_val_c2_st3,'Missing Values','Imputing Data',table_missing_values)
print("Missing Values evaluation Table : Drop missing data")
print(table_missing_values['Random Forest'])