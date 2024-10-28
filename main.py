#!/usr/bin/env python
# coding: utf-8

# #### Suyash Sreekumar
# #### Dataset : Adult Income Dataset
# #### Final Project
# #### Date : 03/18/24 

import pandas as pd
import numpy as np
from config import RANDOM_STATE
from library.evaluation import initialize_evaluation_tables
from library.models import get_base_classifiers,get_weighted_classifiers

#Reading the data from the current directory

df=pd.read_csv('./data/train.csv')

# Creating X and y 

X = df.drop(['income_>50K'], axis =1)
y = df['income_>50K']

random_state = RANDOM_STATE
np.random.seed(random_state)

classifiers = get_base_classifiers(random_state)
classifiers_weights = get_weighted_classifiers(random_state)

# Initialize all tables at once
tables = initialize_evaluation_tables(classifiers)

# Access the tables like this:
table_imbalance = tables['imbalance']
table_missing_values = tables['missing_values']
table_outliers = tables['outliers']

