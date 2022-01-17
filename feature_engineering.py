# %%
from astral import LocationInfo
from astral.sun import sun
import numpy as np
import os
import dash
import dash_core_components as dcc
from dash import html
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import geopy.distance
from scipy import spatial
import pickle

# Configure dataset generation
interpolationMethod = 'sImpute' # "sImpute" or "delete"
weekdayMethod = 'dotw' # 'dotw' or 'wk_wknd'
daylight_switch = True
stationProximity_switch = True

#Perform correlation studies
pca_switch = False  # perform PCA analysis & display results
pearson_switch = False # perform pearson correlation 

# Set load/save path 
dw_directory = "./data" # Set data directory

# %%



# Pull all stations into single df for pre-processing
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

# %% Encode 'weekday' as one-hot weekday encodings
enc = OneHotEncoder(handle_unknown='ignore')
cols_d = np.array(dataset['weekday']).reshape(-1,1)
enc.fit(cols_d)
cols = enc.categories_[0].tolist()
cols_d = pd.DataFrame(enc.transform(cols_d).toarray(), columns=cols)

dataset = dataset.drop(['weekday'], axis=1)

# %% Impute or delete nan rows
if interpolationMethod == "sImpute":
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(dataset)   
    dataset = pd.DataFrame(imp_mean.transform(dataset), columns=dataset.columns)

elif interpolationMethod == 'delete':
    pre_del = len(dataset)
    dataset.dropna(how='any', inplace=True)
    post_del = len(dataset)
    print('Warning: ',(pre_del - post_del)," entries removed from dataset.")
else: 
    print("Error: Interpolation Method not recognised")
    exit

# %% Handling days of the week - one hot encoding per day or week/weekend
if weekdayMethod == 'dotw':
    dataset = pd.concat([dataset, cols_d.reindex(dataset.index)], axis=1)

elif weekdayMethod == 'wk_wknd':
    wk = ['Monday','Tuesday','Wednesday','Thursday']
    wknd = ['Friday','Saturday','Sunday']
    
    wk_array = [0]*len(cols_d)
    for day in wk:
        wk_array = [x + y for x,y in zip(wk_array,cols_d[day])]
    
    wknd_array = [0]*len(cols_d)
    for day in wknd:
        wknd_array = [x + y for x,y in zip(wknd_array,cols_d[day])]

    wk_wnd_df = pd.DataFrame({'week': wk_array, 'weekend': wknd_array})

    dataset = pd.concat([dataset, wk_wkn_df.reindex(dataset.index)], axis=1)
    #cols_wk = pd.DataFrame()

    wk_wknd_plot = False

    if wk_wknd_plot == True:
        x = []
        weekDays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        for day in weekDays:
            idx = [i for i, e in enumerate(cols_d[day]) if e != 0]
            
            x.append(np.mean([dataset['bikes'].values[i]for i in idx]))

        ticks = list(range(0, 7)) 
            # labels = "Mon Tues Weds Thurs Fri Sat Sun".split()
        plt.xticks(ticks, weekDays)
        plt.plot(ticks,x)
else: 
    print("Error: Method for handling days of the week not recognised")
    exit

# %% Calculates sunrise and sunset to give -> is it dark?
if daylight_switch == True:
    city = LocationInfo(39.4502730411, -0.3333629598)

    dk = pd.to_datetime(dataset['timestamp'], unit='s')
    darkness = []
    for i in dk:
        s = sun(city.observer, date=i)
        srise = s['sunrise'].replace(tzinfo=None)
        sset = s['sunset'].replace(tzinfo=None)
        if i < sset and i > srise:
            d = 0
        else:
            d = 1
        if len(darkness) == 0:
            darkness = [d]
        else:
            darkness.append(d)

    dataset['darkness'] = darkness

#%% Distance to another station

if stationProximity_switch == True:
    stations_lat = pd.unique(dataset['latitude'])
    stations_long = pd.unique(dataset['longitude'])

    stations = np.stack((stations_lat, stations_long), axis=1)
    no_s = np.linspace(0,74, 75)

    near = []
    for si1 in no_s:
        distance = []
        for si2 in no_s:
            if si1 == si2:
                continue
            dist = geopy.distance.geodesic(stations[int(si1),:], stations[int(si2),:]).km
            if len(distance) ==0:
                distance = [dist]
            else:
                distance.append(dist)
        nearest = min(distance)
        if len(near) == 0:
            near = [nearest]
        else:
            near.append(nearest)

    def label_dist (row):
        x = near[int(row['station'])-int(201)]
        return x

    dataset['distance'] = dataset.apply (lambda row: label_dist(row), axis=1)

## start and end of work/school
# TBD

# %% Packing and storing datasets

#Divide dataset into individual stations


#count existing files
count = 0
dir = "data/Train/Dataframes/"
for path in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, path)):
        count += 1

file_i = count+1
filename = dir + 'df_' + str(file_i) 

config = {"Interpolation Method":interpolationMethod,
          "Weekday Method"      :weekdayMethod,
          "Light_Dark"          :daylight_switch,
          "Station Proximity"   :stationProximity_switch}

all_stations = dataset
save_list = [config,all_stations]

stations = pd.unique(dataset['station'])

for station in stations:
    idx = dataset.index[dataset['station'] == station].tolist()
    save_list.append(dataset.iloc[idx])

 # %%  


saveFlag = 'y'
df_directory = "./data/Train/DataFrames"
for filename in os.listdir(df_directory):
    
   with open(os.path.join(df_directory, filename), 'rb') as f:
       print(os.path.join(df_directory, filename))
       pickle_list = pickle.load(f)

       if pickle_list[0] == config:

           saveFlag = 'n'

           while True:
                saveFlag = input("Dataframe with current configuration already exists:  " + filename + "| Do you want to overwrite? (y/n) ")
                if saveFlag != "y" or "n":
                    print("Sorry, please enter y/n:")
                    continue
                else:
                    break

           if saveFlag == 'y':
               os.remove(os.path.join(df_directory,filename))

if saveFlag == 'y':
    with open(filename,'wb') as f:        
        pickle.dump(save_list,f)

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
        x = scipy.stats.pearsonr(dataset[i], dataset['bikes'])[0]
        # print(type(pr))
        if len(pr) == 0:
            pr = [x]
        else:    
        # #pr = [pr,x]
            pr.append(x)
    n, bins, patches = plt.hist(dataset['bikes'], 50, density=True, facecolor='g', alpha=0.75)
    fig = plt.barh(datasetb.columns, pr)
    # sns.pairplot(datasetb)

#%%
if pca_switch == True:
    #app = dash.Dash(__name__)
    app = JupyterDash(__name__)


    sl_min = 1
    sl_max = 10

    app.layout = html.Div([
        dcc.Graph(id="graph"),
        html.P("Number of components:"),
        dcc.Slider(
            id='slider',
            min=sl_min, max=sl_max, value=3,
            marks={i: str(i) for i in range(sl_min,sl_max+1)})
    ])

    @app.callback(
        Output("graph", "figure"), 
        [Input("slider", "value")])
    def run_and_plot(n_components):

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(dataset)

        var = pca.explained_variance_ratio_.sum() * 100

        labels = {str(i): f"PC {i+1}" 
                for i in range(n_components)}
        labels['color'] = 'bikes'

        fig = px.scatter_matrix(
            components,
            dimensions=range(n_components),
            labels=labels,
            color=dataset['bikes'],
            title=f'Total Explained Variance: {var:.2f}%')
        fig.update_traces(diagonal_visible=False)
        return fig

    # if __name__ == '__main__':
    #app.run_server(mode='jupyterlab',port=5000)

    app.run_server(mode = 'jupyterlab',port=3050)
# %%
