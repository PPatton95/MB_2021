# %%

from multiprocessing.sharedctypes import Value
import numpy as np
import os
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle

# User-defined
from utilities import data_saver
from fe_utilities import interpolation, weekday_handler, darkness, pca_app, station_proximity

# Save or no
saveMode = True

# Configure dataset generation
interpolationMethod = 'sImpute' # "sImpute" or "delete"
weekdayMethod = 'dotw' # 'dotw' or 'wk_wknd'
daylight_switch = True
stationProximity_switch = True
scale_switch = True

#Perform correlation studies
pca_switch = False  # perform PCA analysis & display results
pearson_switch = False # perform pearson correlation 

# Set load/save path 
dw_directory = "./data" # Set data directory

# Generating test set
# filepath = os.path.join(dw_directory, 'test.csv')
# #if os.path.exists(filepath):
# # Read .txt file
# with open(filepath, 'r') as f:
#     dataset = pd.read_csv(f)       

# Pull all stations into single df for pre-processing
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


# %% Encode 'weekday' as one-hot weekday encodings
enc = OneHotEncoder(handle_unknown='ignore')
days = np.array(dataset['weekday']).reshape(-1,1)
enc.fit(days)
cols = enc.categories_[0].tolist()
days = pd.DataFrame(enc.transform(days).toarray(), columns=cols)

dataset = dataset.drop(['weekday'], axis=1)

dataset_y = pd.DataFrame(dataset['bikes'].copy())
dataset = dataset.drop(['bikes'],axis=1)

# %% Impute or delete nan rows
dataset = interpolation(dataset,interpolationMethod)
dataset_y = interpolation(dataset_y,interpolationMethod)
# %% Handling days of the week - one hot encoding per day or week/weekend
dataset = weekday_handler(dataset,weekdayMethod,days)

# %% Calculates sunrise and sunset to give -> is it dark?
if daylight_switch == True:
    dataset['darkness'] = darkness(dataset)

#%% Distance to another station
if stationProximity_switch == True:
    dataset['distance'] = station_proximity(dataset)

if scale_switch == True:
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    dataset = pd.DataFrame(scaler.transform(dataset), columns=dataset.columns)

# Ask Phil if this is feature engineering ########
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
# categorical_cols = [cname for cname in X_train.columns if
#                     X_train[cname].dtype == "object"]

# # Select numerical columns
# numerical_cols = [cname for cname in X_train.columns if 
#                 X_train[cname].dtype in ['int64', 'float64']]
#    numerical_transformer = SimpleImputer(strategy='constant')

#     # Preprocessing for categorical data
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))
#     ])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numerical_transformer, numerical_cols),
#             ('cat', categorical_transformer, categorical_cols)
#         ])

# %% Packing and storing datasets
config = {"Test"                :False,
          "Interpolation Method":interpolationMethod,
          "Weekday Method"      :weekdayMethod,
          "Light_Dark"          :daylight_switch,
          "Station Proximity"   :stationProximity_switch,
          "Scale Data"          :scale_switch}

xy_dict = {"Name":['X','Y'],"Data":[dataset,dataset_y]}
for i in range(0,2):
    df = xy_dict["Data"][i]
    XorY = xy_dict["Name"][i]

    all_stations = df
    save_list = [config,all_stations]

    if XorY == 'X':
        stations = pd.unique(df['station']) # Return ID of indiviual stations

        for station in stations:
            idx = dataset.index[df['station'] == station].tolist()
            # df = df.drop add a line here for dropping things like station and location
            save_list.append(df.iloc[idx])

        data_saver(config,save_list,XorY)
    elif XorY == 'Y':
        stations = pd.unique(xy_dict["Data"][i-1]['station']) # Return ID of indiviual stations

        for station in stations:
            idx = dataset.index[xy_dict["Data"][i-1]['station'] == station].tolist()
            save_list.append(df.iloc[idx])

        data_saver(config,save_list,XorY)
    else:
        raise ValueError("XorY must be 'X' or 'Y'")
 # %%  

# %% Correlation analysis 
if pearson_switch == True:
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

    pr = []
    for i in dataset.columns:
        x = scipy.stats.pearsonr(dataset[i], dataset_y['bikes'])[0]
        # print(type(pr))
        if len(pr) == 0:
            pr = [x]
        else:    
        # #pr = [pr,x]
            pr.append(x)
    n, bins, patches = plt.hist(dataset_y['bikes'], 50, density=True, facecolor='g', alpha=0.75)
    fig = plt.barh(datasetb.columns, pr)
    # sns.pairplot(datasetb)

#%%
if pca_switch == True:
    pca_app(dataset,2,8)
