#%%
#%load_ext autoreload
#%autoreload 2

## General libraries
import numpy as np # Numpy
import pandas as pd # Pandas
import pandas.plotting
import pickle # Pickles
import os
import logging
import sys
import scipy
import pytz
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

## Usmodel_train_eval
from model_train_eval import bike_trainer
from utilities import data_loader, data_saver

# %% Load Dataset
load_config = {"Test"                :False,
               "Interpolation Method":'sImpute', # "sImpute" or "delete"
               "Weekday Method"      :'dotw',    # 'dotw' or 'wk_wknd'
               "Light_Dark"          :True,
               "Station Proximity"   :True,
               "Scale Data"          :True}

all_stations_X, individual_stations_X = data_loader(load_config,'X')
all_stations_Y, individual_stations_Y = data_loader(load_config,'Y')
datasets = [[all_stations_X,all_stations_Y],
            [individual_stations_X,individual_stations_Y]]
# %% Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

def preprocess(df):
        # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in df.columns if
                        df[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in df.columns if 
                    df[cname].dtype in ['int64', 'float64']]

    #Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    df = df[my_cols].copy()
    return df

# %% PHASE 1A: Individually Trained Models
A_trainX        = []
A_validationX   = []
A_trainY        = []
A_validationY   = []

for i in range(0,len(individual_stations_X)):
    X = individual_stations_X[i]
    Y = individual_stations_Y[i]

    # X = X[~X.isin([np.nan, np.inf, -np.inf]).any(1)]

    atx,avx,aty,avy = train_test_split(X,Y, 
                                       train_size=0.8, 
                                       test_size=0.2,
                                       random_state=0)
    # atx = preprocess(atx)
    print(atx)
    print(aty)

    A_trainX.append(atx)
    A_validationX.append(avx)
    A_trainY.append(aty)
    A_validationY.append(avy)
    atx = atx.reset_index()
    aty = aty.reset_index()
    bike_trainer(atx,aty,model,"station_"+ str(i))

# %% PHASE 1B: Combined Model

btx, bvx, bty, bvy = train_test_split(all_stations_X,
                                    all_stations_Y, 
                                    train_size=0.8, 
                                    test_size=0.2,
                                    random_state=0)

bike_trainer(btx,bty,model,"all_stations")


# # %% Evaluation
# preds = pd.DataFrame(clf.predict(d_X))
# # %%
# preds.index+=1
# # %%

# preds.to_csv('submission.csv', header=['bikes'])
# # %%
