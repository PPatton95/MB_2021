#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:15:11 2022

@author: philippatton
"""

#%% General libraries
import numpy as np # Numpy
import pandas as pd # Pandas
import pandas.plotting
import pickle as pk # Pickles
import datetime as datetime
import os
import logging
import sys
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

# %% Generate Phase1a dataset


dw_directory = "./data"

stations = np.linspace(201, 275, 75)

dataset = pd.DataFrame()

for i in stations:
    # # Read dataset
    filepath = os.path.join(dw_directory, 'Train', 'Train', 'station_' + str(int(i)) + '_deploy.csv')
    #if os.path.exists(filepath):
        # Read .txt file
    with open(filepath, 'r') as f:
         data_v = pd.read_csv(f)
         
    if len(dataset) == 0:
        dataset = data_v
    else:
        dataset = dataset.append(data_v)

        
filepath = os.path.join(dw_directory, 'test.csv')
#if os.path.exists(filepath):
# Read .txt file
with open(filepath, 'r') as f:
    final_test = pd.read_csv(f)       


#%% clean

dataseta = dataset.drop(['weekday'], axis=1)

imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
imp_mean.fit(dataseta)   

dataseta = pd.DataFrame(imp_mean.transform(dataseta), columns=dataseta.columns)

dataseta['weekday'] = dataset['weekday'].reindex().tolist()

dataseta = dataseta.dropna()

# %% Find correlations        
 
#features = pd.DataFrame(data_v.columns)
     
# calculate pearsons coefficient for features
corr = data_v.corr(method = 'pearson')

# plot heatmap
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

fig = ax.get_figure()  

for i in dataset.columns:
    pr = []
    x = scipy.stats.pearsonr(dataseta[i], dataseta['bikes'])[0]
    pr = pr.append(x[0])

    

#%% feature engineering

## is it dark?

city = LocationInfo(39.4502730411, -0.3333629598)

dk = pd.to_datetime(dataseta['timestamp'], unit='s')

for i in dk:
    s = sun(city.observer, date=i)
    
    

## distance to another station

## start and end of work/school


# %% Pipeline

dataset_X = dataseta.drop(['bikes'], axis=1).copy()
dataset_y = dataseta['bikes'].copy()



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
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('scaler', StandardScaler()),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_test)

print('MAE:', mean_absolute_error(y_test, preds))