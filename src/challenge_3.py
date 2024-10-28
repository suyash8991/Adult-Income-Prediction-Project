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
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from library.data_preprocessing import datapreprocessing, standardize
from library.evaluation import evaluate
from src.main import df, random_state, classifiers, table_outliers

np.random.seed(random_state)

# ================CHALLENGE 2 : Outliers================

### Splitting of X into train and validation data along with preprocessing applied

X = df.drop(['income_>50K'], axis =1)
y = df['income_>50K']
X.head()
#Evaluation approach Train Test Split data 
#Using this data now all the three strategies will be evaluated
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=random_state)
X_train,X_val = datapreprocessing(X_train,X_val)

print(X_train.head())


# ================Strategy 1 : Do nothing================
# X and y for the strategy
X_train_c3_st1,X_val_c3_st1=standardize(X_train,X_val)

# Evaluation of strategy 
results = evaluate(classifiers, X_train_c3_st1, X_val_c3_st1, y_train, y_val,'Outliers','No Change',table_outliers)

print("Outliers evaluation Table")
print(table_outliers['Random Forest'])


# ================Strategy 2 : Winsorizing================

# Winsorize each column in the list
columns_to_winsorize = ['capital-gain', 'capital-loss', 'hours-per-week', 'age']

print("Checking 95th-98th percentile for each of the columns to decide where to winsorize the data ")
for column in columns_to_winsorize:

    print(f"\n95th percentile for {column} : {df[column].quantile(0.95)}")
    print(f"96th percentile for {column} : {df[column].quantile(0.96)}")
    print(f"97th percentile for {column} : {df[column].quantile(0.97)}")
    print(f"98th percentile for {column} : {df[column].quantile(0.98)}")

print("\nFor capital gain we are going to cap it at 97th percentile since the no is in between the range")
print("For capital loss, we will take again the 97th percentile to cap it off")
print("For hours per week and age we will tak the standard 95th percentile as the cutoff")

def winsorizing(d_cap):
    # List of columns to winsorize
    df = d_cap.copy()
    columns_to_winsorize = ['capital-gain', 'capital-loss', 'hours-per-week', 'age']

    # Define upper limits for winsorizing based on percentiles
    upper_limits = {
        'capital-gain': 0.03,  # Complement of 97th percentile
        'capital-loss': 0.03,  # Complement of 97th percentile
        'hours-per-week': 0.05,  # Complement of 95th percentile
        'age': 0.05  # Complement of 95th percentile
    }

    # Winsorize each column in the list
    for column in columns_to_winsorize:
        complement_percentile = upper_limits[column]
        print(f"{100 - complement_percentile * 100}th percentile for {column} : {df[column].quantile(1 - complement_percentile)}")
        df[column] = winsorize(df[column], limits=(None, complement_percentile))

    return df

# Applying winsorizing function to both X and X_val

X_train_winsorized=X_train.copy()
X_val_winsorized= X_val.copy()
# Specify the upper limit (95th percentile)
upper_limit = 0.05
X_train_winsorized = winsorizing(X_train_winsorized)
X_val_winsorized=winsorizing(X_val_winsorized)

X_train_winsorized,X_val_winsorized =standardize(X_train_winsorized,X_val_winsorized)

# Evaluate classifiers
results = evaluate(classifiers, X_train_winsorized, X_val_winsorized, y_train, y_val,'Outliers','Winsorizing',table_to_update=table_outliers)
print("Outliers evaluation Table")
print(table_outliers['Random Forest'])


# ================Strategy 3 : Dropping Outliers================

#Need to merge X_train and Y_train so that rows can be dropped 
def merge_df(X_t,y_t):
    # Reset the index of X_train and y_train before merging
    X_t = X_t.reset_index(drop=True)
    y_t = y_t.reset_index(drop=True)

    data_merged=pd.concat([X_t,y_t],axis=1)
    return data_merged

merged_data = merge_df(X_train,y_train)
#Checking size of data
print("X train and y train shape")
print(X_train.shape,y_train.shape)



int_col = ['age', 'hours-per-week', 'capital-gain', 'capital-loss',]

def boxplot(df, cat, ax):
    sns.boxplot(x='income_>50K', y=cat, data=df, ax=ax)
    ax.set_title(f'Boxplot for {cat}')
    ax.set_xlabel(f'Income > 50K')


# Create a 3x3 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))

# Flatten the 2D array of subplots into a 1D array
axes = axes.flatten()

# Iterate over each integer column and plot the boxplot 
for i, col in enumerate(int_col):
    boxplot(df, col, axes[i])

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


print("From above boxplot")
print("The median number of hours worked per week is 40."+
"Many outliers in above upper and below lower quartile."+
"We focus on dropping hours-per-week above 90 since thats too high a no for it be considered normal")

# Carrying out outlier analysis for 'hours-per-week' feature
df_no_outliers = merged_data[merged_data['hours-per-week'] <= 90]

# Carrying out outlier analysis for 'age' feature
df_no_outliers = merged_data[merged_data['age'] < 80]

# Carrying out outlier analysis for 'capital-gain' feature
df_no_outliers = merged_data[merged_data['capital-gain'] < 99999]

# Carrying out outlier analysis for 'capital-loss' feature
df_no_outliers = merged_data[merged_data['capital-loss'] < 2500]

# Carrying out outlier analysis for 'educational-num' feature
df_no_outliers = merged_data[merged_data['educational-num'] > 2]

X_outlier=df_no_outliers.drop('income_>50K',axis=1)
y_outlier=df_no_outliers['income_>50K']

# X and y for strategy 3
X_train_c3_st3,X_val_c3_st3=standardize(X_outlier,X_val)

# Evaluate classifiers
results = evaluate(classifiers, X_train_c3_st3, X_val_c3_st3, y_outlier, y_val,'Outliers','Dropping Outlier',table_outliers)

print("Outliers evaluation Table")
print(table_outliers['Random Forest'])