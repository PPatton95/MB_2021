#%%
# %load_ext autoreload
# %autoreload 2

## General libraries
from distutils.archive_util import make_tarball
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
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

## Usmodel_train_eval
from model_train_eval import bike_inference, bike_trainer
from utilities import data_loader, data_saver

Train_flag = True

features = ['station',
 'latitude',
 'longitude',
 'numDocks',
 'timestamp',
 'weekhour',
 'isHoliday',
 'windMaxSpeed.m.s',
 'windMeanSpeed.m.s',
 'temperature.C',
 'airPressure.mb',
 'bikes_3h_ago',
 'full_profile_3h_diff_bikes',
 'full_profile_bikes',
 'short_profile_3h_diff_bikes',
 'short_profile_bikes',
 'weekend',
 'darkness',
 'distance']

# %% Load Dataset
load_config = {"Features"            :features,
               "Test"                :False,
               "Interpolation Method":'sImpute', # "sImpute" or "delete"
               "Weekday Method"      :'wk_wknd',    # 'dotw' or 'wk_wknd'
               "Light_Dark"          :True,
               "Station Proximity"   :True,
               "Scale Data"          :True}

all_stations_X, individual_stations_X = data_loader(load_config,'X')
all_stations_Y, individual_stations_Y = data_loader(load_config,'Y')
datasets = [[all_stations_X,all_stations_Y],
            [individual_stations_X,individual_stations_Y]]
# %% Define model
<<<<<<< HEAD
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
# Define model


model1 = RandomForestRegressor(n_estimators=100, random_state=0)
model2 = SGDRegressor(max_iter=1000000, tol=1e-3, learning_rate='optimal')
model3 = linear_model.BayesianRidge()
model4 = AdaBoostRegressor(random_state=0, n_estimators=500)
model5 = ExtraTreesRegressor(n_estimators=100, random_state=0)
model6 = BaggingRegressor(base_estimator=SVR(),
                                 n_estimators=10, random_state=0)
models = [model1, model2, model3, model4, model5, model6]

model = model1

#%%
=======
# model = RandomForestRegressor(n_estimators=100, random_state=0)
model = AdaBoostRegressor(random_state=0, n_estimators=500)
#kernel = DotProduct() + WhiteKernel()
#model = GaussianProcessRegressor(kernel=kernel,random_state=0)
>>>>>>> 03c58da4 (report commit)

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

training_ind   = {"predictions":[],"MAE":[]}
validation_ind = {"predictions":[],"MAE":[]}

for i in range(0,len(individual_stations_X)):
    print("Station ",i, " of ", len(individual_stations_X))
    X = individual_stations_X[i]
    Y = individual_stations_Y[i]

    atx,avx,aty,avy = train_test_split(X,Y, 
                                       train_size=0.8, 
                                       test_size=0.2,
                                       random_state=0)
    # atx = preprocess(atx)

    A_trainX.append(atx)
    A_validationX.append(avx)
    A_trainY.append(aty)
    A_validationY.append(avy)

    model_name = "station_"+ str(i)

    if Train_flag == True:
        bike_trainer(atx,aty,model,model_name)

    predictions, MAE = bike_inference(model,model_name,[atx,aty])
    training_ind["predictions"].append(predictions)
    training_ind["MAE"].append(MAE)
    predictions, MAE = bike_inference(model,model_name,[avx,avy])
    validation_ind["predictions"].append(predictions)
    validation_ind["MAE"].append(MAE)      

# %% PHASE 1B: Combined Model

training_all   = {"predictions":[],"MAE":[]}
validation_all = {"predictions":[],"MAE":[]}

btx, bvx, bty, bvy = train_test_split(all_stations_X,
                                    all_stations_Y, 
                                    train_size=0.8, 
                                    test_size=0.2,
                                    random_state=0)


model_name = 'all_stations'

if Train_flag == True:
    bike_trainer(btx,bty,model,model_name)

predictions, MAE = bike_inference(model,model_name,[btx,bty])
training_all["predictions"]=predictions
training_all["MAE"]        =MAE
predictions, MAE = bike_inference(model,model_name,[bvx,bvy])
validation_all["predictions"]=predictions
validation_all["MAE"]        =MAE

# preds.to_csv('submission.csv', header=['bikes'])
# # %%




# %%
print("Ind Stations - Training: ",np.mean(training_ind["MAE"]))
print("All Stations - Training: ",training_all["MAE"])

print("Ind Stations - Validation: ",np.mean(validation_ind["MAE"]))
print("All Stations - Validation: ",validation_all["MAE"])
# %%

load_config = {"Test"                :True,
               "Interpolation Method":'sImpute', # "sImpute" or "delete"
               "Weekday Method"      :'dotw',    # 'dotw' or 'wk_wknd'
               "Light_Dark"          :True,
               "Station Proximity"   :True,
               "Scale Data"          :True}


all_stations_X, individual_stations_X = data_loader(load_config,'X')
all_stations_X = all_stations_X.drop(['Id'],axis=1)

test_all   = {"predictions":[],"MAE":[]}
test_ind   = {"predictions":[],"MAE":[]}
for i in range(0,len(individual_stations_X)):
    model_name = "station_"+ str(i)

    atx = individual_stations_X[i].drop(['Id'],axis=1)
    aty = pd.DataFrame(np.zeros(len(atx)))

    predictions, MAE = bike_inference(model,model_name,[atx,aty])
    for p in predictions:
        test_ind["predictions"]= test_ind["predictions"] + [p]
    test_ind["MAE"].append(MAE)

model_name = "all_stations"
btx = all_stations_X
bty = np.zeros(len(btx))
predictions, MAE = bike_inference(model,model_name,[btx,bty])
test_all["predictions"]=predictions
test_all["MAE"]        =MAE
# %%

ind_test = pd.DataFrame({"bikes":test_ind["predictions"]})
ind_test = ind_test.round()
ind_test.index+=1
ind_test.to_csv('data/USER/Submissions/' + 'reduced_individual_model' +'submission.csv',header=['bikes'])
# %%

all_test = pd.DataFrame({"bikes":test_all["predictions"]})
all_test = all_test.round()
all_test.index+=1
all_test.to_csv('data/USER/Submissions/' + 'reduced_all_stations' +'submission.csv',header=['bikes'])

# %%
