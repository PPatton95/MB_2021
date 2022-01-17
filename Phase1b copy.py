#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:15:11 2022

@author: philippatton
"""
#%%

## General libraries
## General libraries
import numpy as np # Numpy
import pandas as pd # Pandas
import pandas.plotting
import pickle # Pickles
import datetime as datetime
import os
import logging
import sys
import astral
import scipy
import pytz

#import heatmapz
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from astral import LocationInfo
from astral.sun import sun

# %% Generate Phase1a dataset


dw_directory = "./data"
df_directory = "./data/Train/DataFrames"

interpolationMethod = 'sImpute' # "sImpute" or "delete"
weekdayMethod = 'dotw' # 'dotw' or 'wk_wknd'
daylight_switch = True
stationProximity_switch = True

load_config = {"Interpolation Method":interpolationMethod,
          "Weekday Method"      :weekdayMethod,
          "Light_Dark"          :daylight_switch,
          "Station Proximity"   :stationProximity_switch}

for filename in os.listdir(df_directory):
   with open(os.path.join(df_directory, filename), 'rb') as f:
       print(os.path.join(df_directory, filename))
       pickle_list = pickle.load(f)

       if pickle_list[0] == load_config:
           print("TRUE")
           all_stations = pickle_list[1]
           ind_stations = []
           for station in pickle_list[2:]:
               ind_stations.append(station)
       else:
            print("FALSE")

datasetb = all_stations

dataset_X = datasetb.drop(['bikes'], axis=1).copy()
dataset_y = datasetb['bikes'].copy()



# Break off validation set from training data
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, 
        
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train.columns if
                    X_train[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train.columns if 
                X_train[cname].dtype in ['int64', 'float64']]

#Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train[my_cols].copy()
#X_valid = X_valid_full[my_cols].copy()
X_test = X_test[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#X_train.isna().any()

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('scaler', StandardScaler()),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_test)

print('MAE:', mean_absolute_error(y_test, preds))
# %%
