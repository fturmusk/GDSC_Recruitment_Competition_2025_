from src.preprocess import load_data, column_name
from src.preprocess import cont_nan_values, nan_values_transformer
from src.preprocess import onehot_encoder, sqrt_high_values
from src.exploration import y_train_describe, train_test_visualisation
from src.exploration import train_info, numeric_object_nan, object_unic
from src.visualisation import train_histplot,train_histplot, train_subplot, train_contplot
from src.model import best_model_,test_predict_and_submit

def main():
    path1 = r"C:\Users\batoc\Desktop\Github\1\data\data_train.csv"
    path2 = r"C:\Users\batoc\Desktop\Github\1\data\data_test.csv"
    
    y_train, df_test, df_train, df, df__test = load_data(path1,path2)
    nan_values_count, numerics_columns, object_columns = column_name(df_train)

    y_train_describe(df_train)
    train_test_visualisation(df_train, df_test)
    train_info(df_train)
    
    nan_values_count = cont_nan_values(df_train)

    numeric_object_nan(nan_values_count, numerics_columns, object_columns)
    object_unic(object_columns, df_train)

    train_transform, test_transform = nan_values_transformer(df_train, df_test, numerics_columns, object_columns, y_train)

    train_histplot(numerics_columns,train_transform)
    train_subplot(numerics_columns, train_transform)
    train_contplot(object_columns, train_transform, df)

    train_final, test_final = onehot_encoder(object_columns,train_transform, test_transform )
    train_final, test_final = sqrt_high_values(train_final,test_final)

    best_model = best_model_(train_final,y_train)

    test_predict_and_submit(best_model,train_final,y_train, test_final,df__test)

if __name__ == "__main__":
    main()
