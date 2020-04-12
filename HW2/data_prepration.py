from HW2.data_technical import *
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def main():
    # technical stuff : the functions that handle them located in data_technical
    df = load_data(DATA_PATH)
    # categorized nominal attributes to int
    df = categorize_data(df)

    # export raw data to csv files
    x_train, x_val, x_test, y_train, y_val, y_test = split_database(df,
                                                                    global_test_size,
                                                                    global_validation_size)
    export_to_csv(PATH, x_train, x_val, x_test, y_train, y_val, y_test,
                  prefix="raw")

    # data cleansing
    #there are some data which is negative and thus incorrect such as negative expense, avg income and etc..
    x_train, x_val, x_test = change_negative_to_null(x_train, x_val, x_test)
    # outliers detection
    x_train, x_val, x_test = remove_outlier_values(x_train, x_val, x_test, global_z_threshold)

    # imputation
    #TODO: implement imputation function

    # normalization (scaling)
    x_train, x_val, x_test = normalization(x_train, x_val, x_test)

    #feature selection
    #TODO: filter method+ wrapper method

    #TODO: wrapper method

    #save the data
    #TODO:

def normalization(x_train: DataFrame, x_val: DataFrame, x_test: DataFrame):
    scale_std = StandardScaler()
    # scale the values to fit range
    scale_min_max = MinMaxScaler(feature_range=(-1, 1))
    local_uniform_features = [f for f in uniform_features if
                              f not in nominal_features]
    x_train[local_uniform_features] = scale_min_max.fit_transform(x_train[local_uniform_features])
    x_val[local_uniform_features] = scale_min_max.transform(x_val[local_uniform_features])
    x_test[local_uniform_features] = scale_min_max.transform(x_test[local_uniform_features])

    local_non_uniform = [f for f in labeless_features if f not in uniform_features and f not in nominal_features]

    x_train[local_non_uniform] = scale_std.fit_transform(x_train[local_non_uniform])
    x_val[local_non_uniform] = scale_std.transform(x_val[local_non_uniform])
    x_test[local_non_uniform] = scale_std.transform(x_test[local_non_uniform])
    return x_train, x_val, x_test


def remove_outlier_values(x_train: DataFrame, x_val: DataFrame, x_test: DataFrame,
                    z_threshold: float):
    mean_train = x_train[normal_features].mean()
    std_train = x_train[normal_features].std()

    dist_train = (x_train[normal_features] - mean_train) / std_train
    dist_val = (x_val[normal_features] - mean_train) / std_train
    dist_test = (x_test[normal_features] - mean_train) / std_train

    data_list = [x_train, x_val, x_test]
    dist_list = [dist_train, dist_val, dist_test]

    for feature in normal_features:
        for df, dist in zip(data_list, dist_list):
            for i in dist[feature].loc[(dist[feature] > z_threshold) | (dist[feature] < -z_threshold)].index:
                df.at[i, feature] = np.nan

    return x_train, x_val, x_test


def change_negative_to_null(x_train: DataFrame, x_val: DataFrame,
                   x_test: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    for feature in numerical_features:
        x_train.loc[(~x_train[feature].isnull()) & (
                x_train[feature] < 0), feature] = np.nan
        x_val.loc[(~x_val[feature].isnull()) & (
                x_val[feature] < 0), feature] = np.nan
        x_test.loc[(~x_test[feature].isnull()) & (
                x_test[feature] < 0), feature] = np.nan
    return x_train, x_val, x_test





if __name__ == '__main__':
    main()
