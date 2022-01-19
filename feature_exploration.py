# %%
# %load_ext autoreload
# %autoreload 2

from multiprocessing.sharedctypes import Value
import numpy as np
import os
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
import pickle

# User-defined
from utilities import data_saver
from fe_utilities import interpolation, weekday_handler, darkness, pca_app, station_proximity

# Save or no
testFlag = False

# Configure dataset generation
interpolationMethod = 'sImpute' # "sImpute" or "delete"
weekdayMethod = 'wk_wknd' # 'dotw' or 'wk_wknd'
daylight_switch = True
stationProximity_switch = False
scale_switch = True

#Perform correlation studies
pca_switch = False  # perform PCA analysis & display results
pearson_switch = False # perform pearson correlation 

# Set load/save path 
dw_directory = "./data" # Set data directory

stations = np.linspace(201, 275, 75)
dataset = pd.DataFrame()

for i in stations:
    filepath = os.path.join(dw_directory, 'Train', 'Train', 'station_' + str(int(i)) + '_deploy.csv')

    with open(filepath, 'r') as f:
        data_v = pd.read_csv(f)
         
        if len(dataset) == 0:
            dataset = data_v
        else:
            dataset = dataset.append(data_v)

bikes = pd.DataFrame(dataset['bikes'].copy())
dataset = dataset.drop(['bikes'],axis=1)
# %%
redundant_columns = ['year','month','precipitation.l.m2']
dataset = dataset.drop(redundant_columns,axis =1)
less_significant_columns = ['relHumidity.HR','windDirection.grades','hour','day']
dataset = dataset.drop(less_significant_columns,axis=1)

enc = OneHotEncoder(handle_unknown='ignore')
days = np.array(dataset['weekday']).reshape(-1,1)
enc.fit(days)
cols = enc.categories_[0].tolist()
days = pd.DataFrame(enc.transform(days).toarray(), columns=cols)

dataset = dataset.drop(['weekday'], axis=1)

# %% Impute or delete nan rows

dataset = interpolation(dataset, interpolationMethod)
bikes = interpolation(bikes, interpolationMethod)
#  Handling days of the week - one hot encoding per day or week/weekend
dataset = weekday_handler(dataset,weekdayMethod,days)

#  Calculates sunrise and sunset to give -> is it dark?
if daylight_switch == True:
    dataset['darkness'] = darkness(dataset)

# Distance to another station
if stationProximity_switch == True:
    dataset['distance'] = station_proximity(dataset)

if scale_switch == True:
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    dataset = pd.DataFrame(scaler.transform(dataset), columns=dataset.columns)


columns = dataset.columns.tolist()
print(len(columns))

# %%

plotCharts = False
if plotCharts == True:
    for column in columns:
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir,'data/USER/Figures/')
        fname = 'feature_' + column + '.eps'

        x = dataset[column].values
        y = bikes.values
        plt.figure()
        plt.scatter(x,y,marker='.')
        plt.xlabel(column,fontsize = 20)
        plt.ylabel(r'$N_{bikes}$',fontsize = 20)
        # Show/save figure as desired.
        plt.savefig(results_dir + fname,bbox_inches='tight')
        plt.show()


# %%
pca_switch = False
if pca_switch == True:
    pca_app(dataset,bikes,2,10)

# %%
pls = PLSRegression(n_components=1)
plotPLS = True
if plotPLS == True:
    for column in columns:
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir,'data/USER/Figures/')
        x = dataset[column].values.reshape(-1, 1)
        y = bikes["bikes"]#.reshape(-1, 1)

        pls.fit(x, y)

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir,'data/USER/Figures/')
        fname = 'PLS_feature_' + column + '.eps'
        
        plt.figure()
        
        plt.scatter(pls.transform(x), y, alpha=0.3, label="ground truth")
        plt.scatter(pls.transform(x), pls.predict(x), alpha=0.3, label="predictions")
        
        # plt.scatter(x,y,marker='.',color='k')
        plt.xlabel(column,fontsize = 20)
        plt.ylabel(r'$N_{bikes}$',fontsize = 20)
        # Show/save figure as desired.
        plt.savefig(results_dir + fname,bbox_inches='tight')
        plt.show()
# %%
def plot_corr_map(station_id, ax, labels=True):
    df = pd.read_csv(get_train_path(station_id))
    df = df.drop(['station', 'weekday'], axis=1)

    s = df.corrwith(df['bikes'])
    if not labels: s = np.array(s)
    df = pd.DataFrame({'bikes':s})

    ax.set_title('Station '+str(station_id))
    sns.heatmap(df, annot=True, ax=ax)
    
fig, ax = plt.subplots(1, 3, figsize=(15,10)) 

## Investigate correlation between features and target variable for specific stations
plot_corr_map(201, ax=ax[0])
plot_corr_map(202, ax=ax[1], labels=False)
plot_corr_map(STATION_ID, ax=ax[2], labels=False)