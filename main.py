#!/usr/bin/env python
# coding: utf-8

# #### Suyash Sreekumar
# #### Dataset : Adult Income Dataset
# #### Final Project
# #### Date : 03/18/24 

import pandas as pd
import numpy as np
from config import RANDOM_STATE


#Reading the data from the current directory

df=pd.read_csv('./data/train.csv')

# Creating X and y 

X = df.drop(['income_>50K'], axis =1)
y = df['income_>50K']

random_state = RANDOM_STATE
np.random.seed(random_state)

