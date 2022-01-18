
#%%
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


# %%
no_stations = np.linspace(201,275, 75)

dw_directory = "./data"
dw_directory = "/Users/philippatton/Documents/Data science/morebikes2021/"

dataset = pd.DataFrame()

for i in no_stations:
    # # Read dataset
    filepath = os.path.join(dw_directory, 'Train', 'Train', 'station_' + str(int(i)) + '_deploy.csv')
    #if os.path.exists(filepath):
        # Read .txt file
    with open(filepath, 'r') as f:
         data_v = pd.read_csv(f)
         sns.lineplot(data = dataset, x = 'day', y = 'bikes')
    if len(dataset) == 0:
        dataset = data_v
    else:
        dataset = dataset.append(data_v)

#%%

sts = {}
for i in no_stations:
    # # Read dataset
    filepath = os.path.join(dw_directory, 'Train', 'Train', 'station_' + str(int(i)) + '_deploy.csv')
    #if os.path.exists(filepath):
        # Read .txt file
    with open(filepath, 'r') as f:
         data_v = pd.read_csv(f)
         
         hours = np.linspace(0,23,24)
         st = int(i)

         sts[i] = pd.DataFrame()
         for h in hours:
             hrs = data_v[data_v['hour'] == h]
             mean = np.array(hrs.mean())
             mean = pd.DataFrame(mean, columns=data_v.columns)

             if len(sts[i]) == 0:
                 sts[i] = pd.DataFrame(mean)
             else:
                 sts[i] = sts[i].append(mean, ignore_index=True)

#%%
sts[i].head()

#%%
sns.lineplot(data = dataset, x = 'day', y = 'bikes')
# %%
