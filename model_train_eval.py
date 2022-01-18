import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

def bike_trainer(df_X,df_Y,model,name):
    
    # MOVED TO FEATURE ENGINEERING
    # #Keep selected columns only
    # my_cols = categorical_cols + numerical_cols
    # X_train = X_train[my_cols].copy()
    # #X_valid = X_valid_full[my_cols].copy()
    # X_test = X_test[my_cols].copy()
    if 'sklearn.ensemble._forest.RandomForestRegressor' in str(type(model)):
        # Bundle preprocessing and modeling code in a pipeline
        clf = Pipeline(steps=[('scaler', StandardScaler()),('model', model)])
        # Preprocessing of training data, fit model 

        clf.fit(df_X, df_Y)

        model_saver(clf,name,'sklearn_randomforest')
        # Preprocessing of validation data, get predictions
    else: 
        raise ValueError("I don't know what this is yet")


def bike_inference(model,model_type,data):
    
    if model_type == "RandomForestRegressor":

        prediction  = model.predict(data)

    
    return mean_absolute_error(result,prediction)


