# %%

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

# Convert weekdays to numeric
# dotw = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# for i in range(0,len(dataset)):
#     try:
#         dataset['weekday'][i] = dotw.index(dataset['weekday'].values[i])
#     except:
#         if 0 <= dataset['weekday'].values[i] < 7:
#             continue
#         else:
#             print('Error')
#             break
# %%
#dataseta = dataset.drop(['weekday'], axis=1)
# One-hot weekday encodings
enc = OneHotEncoder(handle_unknown='ignore')
cols_d = np.array(dataset['weekday']).reshape(-1,1)
enc.fit(cols_d)
cols = enc.categories_[0].tolist()
cols_d = pd.DataFrame(enc.transform(cols_d).toarray(), columns=cols)

dataset = dataset.drop(['weekday'], axis=1)

dataset = pd.concat([dataset, cols_d.reindex(dataset.index)], axis=1)
#dataset =  pd.concat([dataset, cols_d], axis=1)

# %%

imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
imp_mean.fit(dataset)   

dataset = pd.DataFrame(imp_mean.transform(dataset), columns=dataset.columns)

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
