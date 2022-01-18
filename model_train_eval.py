import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from utilities import model_saver, model_loader

def bike_trainer(df_X,df_Y,model,name):
    
    # MOVED TO FEATURE ENGINEERING
    # #Keep selected columns only
    # my_cols = categorical_cols + numerical_cols
    # X_train = X_train[my_cols].copy()
    # #X_valid = X_valid_full[my_cols].copy()
    # X_test = X_test[my_cols].copy()
    # model1 = RandomForestRegressor(n_estimators=100, random_state=0)
    # model2 = SGDRegressor(max_iter=1000000, tol=1e-3, learning_rate='optimal')
    # model3 = linear_model.BayesianRidge()
    # model4 = AdaBoostRegressor(random_state=0, n_estimators=500)
    # model5 = ExtraTreesRegressor(n_estimators=100, random_state=0)
    # model6 = BaggingRegressor(base_estimator=SVR(),
    #                                  n_estimators=10, random_state=0)


    test_string = str(type(model)).lower

    if 'sklearn' in test_string:
        model_name = test_string[test_string.rfind('.')+1:-2]
        # Bundle preprocessing and modeling code in a pipeline
        clf = Pipeline(steps=[('model', model)])

        # Preprocessing of training data, fit model 
        df_Y = np.array(df_Y['bikes'])

        clf.fit(df_X, df_Y)

        model_saver(clf,'sklearn_'+model_name,name)
    else: 

        raise ValueError("I don't know what this is yet")


def bike_inference(model,model_name,data):
    model_type = type(model)
    test_string = str(model_type).lower()
    
    if 'sklearn' in test_string:
        model_name = test_string[test_string.rfind('.')+1:-2]

        model = model_loader(model,'sklearn_'+model_name,model_name)
    
        prediction  = model.predict(data[0])
    
    return prediction, mean_absolute_error(data[1],prediction)


