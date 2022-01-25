
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
         #print(i)
         #sns.lineplot(data = data_v, x = 'hour', y = 'bikes')
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
data_bike = data_v.copy()
data_bike['plot'] = 'bikes'
data_bike2 = data_v.copy()
data_bike2['plot'] = 'bikes_3h_ago'
data_bike['bikes'] = data_bike['bikes_3h_ago']

data_b = data_bike.append(data_bike2)
data_b = data_b.reset_index()

ax = sns.lineplot(data = data_b, x = 'hour', y = 'bikes', hue = 'plot')
#ax = sns.lineplot(data = data_v, x = 'hour', y = 'bikes_3h_ago')
plt.legend()
plt.show()
# %%

import geopandas
import folium

stations_lat = pd.unique(dataset['latitude'])
stations_long = pd.unique(dataset['longitude'])
stations_no = pd.DataFrame(pd.unique(dataset['station']), columns=['station'])

stations = np.stack((stations_lat, stations_long), axis=1)
stations = pd.DataFrame(stations, columns = ['latitude', 'longitude'])

locations = stations[['latitude', 'longitude']]
locationlist = locations.values.tolist()

map = folium.Map(location = [39.4502730411,-0.333362], tiles='OpenStreetMap' , zoom_start = 9)

for point in range(0, len(stations)):
    folium.Marker(locationlist[point], popup=stations_no['station'][point]).add_to(map)

map
# %%
import tsfresh 

#data_v['timestamp'].plot(subplots=True, sharex=True, figsize=(10,10))


data_t = dataset[['station', 'hour', 'bikes_3h_ago']]
data_t = data_v[['day', 'hour', 'bikes_3h_ago']]
data_t = data_t.dropna()
#data_t['timestamp'] = pd.to_datetime(data_t['timestamp'], unit='s')
#%%
from tsfresh import extract_features
extracted_features = extract_features(data_t, column_id="day", column_sort="hour", n_jobs=0)

# %%pip
from tsfresh import extract_relevant_features

features_filtered_direct = extract_relevant_features(data_t, data_t['bikes'],
                                                     column_id='station', column_sort='timestamp')


# %%
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, ComprehensiveFCParameters
settings = MinimalFCParameters()
extracted_features = extract_features(data_t, column_id="day", column_sort="hour", default_fc_parameters=settings, n_jobs=0)
# %%
import tsfresh 
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()
# %%
from tsfresh import extract_features
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
# %%
