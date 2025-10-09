import numpy as np
import pandas as pd

def y_train_describe(y_train):
    print("desciption of elements for y_train \n")
    print(y_train.describe())

def train_test_visualisation(df_train, df_test):
    print("Visualisation for Train")
    print(df_train.head())
    print("Visualisation for Test")
    print(df_test.head())

def train_info(df_train):
    print("information about feacture")
    print(df_train.info())

def numeric_object_nan(nan_values_count, numerics_columns, object_columns):
    i = 0
    for key,val in nan_values_count.items():
        if key in numerics_columns:
            i+=1
            print(key,"-------", val)
    print("Numerical Nan Rows number : ",i)

    i=0
    print("\nCount Nan for type Object \n")
    for key,val in nan_values_count.items():
        if key in object_columns:
            i+=1
            print(key,"-------", val)
    print("Categorical Nan Rows number : ",i)

def object_unic(object_columns, df_train):
    for col in object_columns:
        print("Unique values for rows ",col," : ",df_train[col].unique(),"\n","number of unique values : ",len(df_train[col].unique()),"\n")



