import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

def bike_trainer(df,model):
    

    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train.columns if
                        X_train[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train.columns if 
                    X_train[cname].dtype in ['int64', 'float64']]

    #Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train[my_cols].copy()
    #X_valid = X_valid_full[my_cols].copy()
    X_test = X_test[my_cols].copy()

    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('scaler', StandardScaler()),('model', model)])
    # Preprocessing of training data, fit model 
    clf.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    
    dump(clf, 'filename.joblib') 
    output = {"Training Results":y_train,"Training Predictions":pred_train,
              "Validation Results":y_test,"Validation Predictions":pred_test}
    return output

def eval(model,model_type,data):
    
    if model_type == "RandomForestRegressor":

        prediction  = model.predict(data)

    
    return mean_absolute_error(result,prediction)


