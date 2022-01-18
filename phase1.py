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

for df in datasets:
    # Break off validation set from training data
    X_train, X_test, y_train, y_test = train_test_split(dataset_X,
                                                        dataset_y, 
                                                        train_size=0.8, 
                                                        test_size=0.2,
                                                        random_state=0)

output_keys = ["Training Results","Training Predictions","Validation Results","Validation Predictions"]
# %% PHASE 1A: Individually Trained Models
ind_outputs = []
ind_all_outputs = {output_keys[0]:[],output_keys[1]:[],
                   output_keys[2]:[],output_keys[3]:[]}

for d_set in individual_stations:
    output = (bike_trainer(d_set,model))
    ind_outputs.append(output)
    for key in output_keys:
        ind_all_outputs[key].extend(output[key])

print("Training Error | Individual Models : MAE = ",
    mean_absolute_error(ind_all_outputs["Training Results"],
                        ind_all_outputs["Training Predictions"]))

print("Validation Error | Individual Models : MAE = ",
    mean_absolute_error(ind_all_outputs["Validation Results"],
                        ind_all_outputs["Validation Predictions"]))

# %% PHASE 1B: Combined Model
all_outputs = bike_trainer(all_stations,model)

print("Training Error | Combined Models : MAE = ",
    mean_absolute_error(all_outputs["Training Results"],
                        all_outputs["Training Predictions"]))

print("Validation Error | Combined Model : MAE = ",
    mean_absolute_error(all_outputs["Validation Results"],
                        all_outputs["Validation Predictions"]))

# %% Evaluation
preds = pd.DataFrame(clf.predict(d_X))
# %%
preds.index+=1
# %%

preds.to_csv('submission.csv', header=['bikes'])
# %%
