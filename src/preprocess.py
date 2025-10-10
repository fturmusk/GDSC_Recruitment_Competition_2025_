import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import random
import math
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

def load_data(path, path2):
    df = pd.read_csv(path)
    df__test = pd.read_csv(path2)

    y_train = df.iloc[:, -1]
    df_train = df.iloc[:, 1:-1]
    df_test = df__test.iloc[:, 1:]

    return y_train, df_test, df_train, df, df__test

def column_name(df_train):

    columns_names = df_train.columns.tolist()
    
    object_columns = (df_train.select_dtypes(include = "object").columns).tolist()
    numerics_columns = (df_train.select_dtypes(exclude = "object").columns).tolist()

    return columns_names, object_columns, numerics_columns

def cont_nan_values(df_train):

    nan_columnns = [col for col in df_train.columns if df_train[col].isna().any()]
    count_nan = [df_train[col].isna().sum() for col in nan_columnns ]
    nan_values_count = dict(zip(nan_columnns, count_nan))
    return nan_values_count

def nan_values_transformer(df_train, df_test, numerics_columns, object_columns, y_train):
    print(object_columns)
    numerical_transformers = SimpleImputer(strategy= "median")
    categorical_transformers = Pipeline(steps=[("imputer", SimpleImputer(strategy = "most_frequent"))])

    preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformers, numerics_columns),
    ("cat", categorical_transformers, object_columns)
    ])

    train_transform = preprocessor.fit_transform(df_train)
    feature_names = preprocessor.get_feature_names_out()
    train_transform = pd.DataFrame(train_transform, columns = (numerics_columns + object_columns))

    test_transform = preprocessor.fit_transform(df_test)
    feature_names = preprocessor.get_feature_names_out()
    test_transform = pd.DataFrame(test_transform, columns = (numerics_columns + object_columns))

    df_concat = pd.concat([train_transform, y_train], axis = 1)
    df_concat.columns = df_concat.columns.astype(str)
    df_concat = df_concat.dropna()
    y_train = df_concat.iloc[:, -1]
    train_transform = df_concat.iloc[:, :-1]

    train_transform[numerics_columns] = train_transform[numerics_columns].astype("float64")
    test_transform[numerics_columns] = test_transform[numerics_columns].astype("float64")

    
    return train_transform, test_transform, y_train

def onehot_encoder(object_columns,train_transform, test_transform ):
    

    train_transform.drop(object_columns[1],axis = 1, inplace = True)
    test_transform.drop(object_columns[1],axis = 1, inplace = True)

    object_columns.pop(1)

    hotencoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    train_hot_transform = pd.DataFrame(hotencoder.fit_transform(train_transform[object_columns]))
    test_hot_transform = pd.DataFrame(hotencoder.fit_transform(test_transform[object_columns]))

    train_hot_transform.index = train_transform.index
    test_hot_transform.index = test_transform.index

    train_transform.drop(object_columns,axis = 1, inplace = True)
    test_transform.drop(object_columns,axis = 1, inplace = True)

    train_final = pd.concat([train_transform,train_hot_transform], axis = 1)
    test_final = pd.concat([test_transform,test_hot_transform], axis = 1)

    train_final.columns = train_final.columns.astype(str)
    test_final.columns = test_final.columns.astype(str)


    new_featurs_name = []
    names = "q,"
    nam = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z Ä Ö Ü 1 2 3 4 5 6 7 8 9 0 ß"

    for name in range(len(train_final.columns)):
        if len(new_featurs_name) < len(nam.split()):
            new_featurs_name.append(nam.split()[name])
        else:
            if len(new_featurs_name) <= len(train_final.columns):
                names = nam.split()[random.randint(0, len(nam.split())-1)] + nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]+ nam.split()[random.randint(0, len(nam.split())-1)]
            if names in new_featurs_name:
                name = name -1
            else :
                new_featurs_name.append(names)
    train_final.columns = new_featurs_name
    test_final.columns = new_featurs_name

    return train_final, test_final

def sqrt_high_values(train_final,test_final):

    for val in train_final.columns:
        if np.max(train_final[val]) > 11 or np.min(train_final[val]) < -11:
            for i in range(len(train_final[val])-1):
                if train_final[val].iloc[i] < 0:
                    train_final[val].iloc[i] = train_final[val].iloc[i] * -1
                    train_final[val].iloc[i] = math.sqrt(train_final[val].iloc[i])
                    train_final[val].iloc[i] = train_final[val].iloc[i] * -1
                else :
                    train_final[val].iloc[i] = math.sqrt(train_final[val].iloc[i])

    for val in test_final.columns:
        if np.max(test_final[val]) > 11 or np.min(test_final[val]) < -11:
            for i in range(len(test_final[val]) - 1):
                if test_final[val].iloc[i] < 0:
                    test_final[val].iloc[i] = test_final[val].iloc[i] * -1
                    test_final[val].iloc[i] = math.sqrt(test_final[val].iloc[i])
                    test_final[val].iloc[i] = test_final[val].iloc[i] * -1
                else :
                    test_final[val].iloc[i] = math.sqrt(test_final[val].iloc[i])
    
    return train_final, test_final