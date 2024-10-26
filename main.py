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



#Reading the data from the current directory

df=pd.read_csv('./train.csv')

# Creating X and y 

X = df.drop(['income_>50K'], axis =1)
y = df['income_>50K']

random_state = 42
np.random.seed(random_state)

