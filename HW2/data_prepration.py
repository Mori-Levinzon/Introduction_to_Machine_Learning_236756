from HW2.data_technical import *
from HW2.selected_features import *
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from impyute import imputation
from csv import writer


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
    # there are some data which is negative and thus incorrect such as negative expense, avg income and etc..
    x_train, x_val, x_test = change_negative_to_null(x_train, x_val, x_test)

    # outliers detection
    x_train, x_val, x_test = remove_outlier_values(x_train, x_val, x_test, global_z_threshold)

    # imputation
    x_train, x_val, x_test = imputations(x_train, x_val, x_test, y_train, y_val, y_test)

    # normalization (scaling)
    x_train, x_val, x_test = normalization(x_train, x_val, x_test)

    # feature selection

    # filter method
    selected_numerical_features_by_variance = variance_filter(x_train[numerical_features], y_train,
                                                              global_variance_threshold)
    selected_features_by_variance = selected_numerical_features_by_variance + nominal_features
    print("the not chosen features are: {}".format(
        [f for f in numerical_features if f not in selected_features_by_variance]))
    x_train = x_train[selected_features_by_variance]
    x_val = x_val[selected_features_by_variance]
    x_test = x_test[selected_features_by_variance]

    # wrapper method
    selected_features_by_mi = apply_mi_wrapper_filter(x_train, y_train)
    x_train = x_train[selected_features_by_mi]
    x_val = x_val[selected_features_by_mi]
    x_test = x_test[selected_features_by_mi]
    export_to_csv(PATH, x_train, x_val, x_test, y_train, y_val, y_test, prefix="fixed")

    # save the data
    final_selected_features = x_train.columns.values.tolist()
    export_selected_features(SELECTED_FEATURES_PATH, final_selected_features)


def export_selected_features(filename: str, selected_features_list: list):
    with open(filename, 'w') as csv_file:
        wr = writer(csv_file)
        wr.writerow(selected_features_list)


def get_features_correlation(data: DataFrame,
                             features_correlation_threshold: float):
    correlation_dict = defaultdict(list)
    for f1 in data.columns:
        for f2 in data.columns:
            if f2 == f1:
                continue
            correlation = data[f1].corr(data[f2], method='pearson')  # calculating pearson correlation
            if abs(correlation) >= features_correlation_threshold:
                correlation_dict[f1].append(f2)
    return correlation_dict


def fill_feature_correlation(data: DataFrame, correlation_dict: dict):
    for f1 in correlation_dict.keys():
        for f2 in correlation_dict[f1]:
            coef_val = (data[f2] / data[f1]).mean()
            # fill values for data set
            other_approximation = data[f1] * coef_val
            data[f2].fillna(other_approximation, inplace=True)


def closest_fit_imputation(ref_data: DataFrame, data_to_fill: DataFrame):
    for index, row in data_to_fill[data_to_fill.isnull().any(axis=1)].iterrows():
        print("index - ", index, ": row - ", row)
        row.fillna(ref_data.iloc[closest_fit(ref_data, row, nominal_features, numerical_features)], inplace=True)


def imputations(x_train: DataFrame, x_val: DataFrame, x_test: DataFrame,
                y_train: DataFrame, y_val: DataFrame, y_test: DataFrame):
    train = x_train.assign(Vote=y_train.values)
    val = x_val.assign(Vote=y_val.values)
    test = x_test.assign(Vote=y_test.values)

    # fill missing values by using information from correlated features
    correlation_dict_train = get_features_correlation(train,
                                                      global_correlation_threshold)

    fill_feature_correlation(train, correlation_dict_train)
    fill_feature_correlation(val, correlation_dict_train)
    fill_feature_correlation(test, correlation_dict_train)

    # fill missing data using closest fit
    print("closest fit - train")
    closest_fit_imputation(train.dropna(how='any'), train)
    print("closest fit - validation")
    closest_fit_imputation(val.dropna(how='any'), val)
    print("closest fit - test")
    closest_fit_imputation(test.dropna(how='any'), test)

    # fill normal distributed features using EM algorithm
    train_after_em = imputation.cs.em(np.array(train[normal_features]), loops=50, dtype='cont')
    train.loc[:, normal_features] = train_after_em

    test_after_em = imputation.cs.em(np.array(test[normal_features]), loops=50, dtype='cont')
    test.loc[:, normal_features] = test_after_em

    val_after_em = imputation.cs.em(np.array(val[normal_features]), loops=50, dtype='cont')
    val.loc[:, normal_features] = val_after_em

    # fill using statistics
    # for numerical feature filling by median
    train[numerical_features] = train[numerical_features].fillna(train[numerical_features].median(), inplace=False)
    val[numerical_features] = val[numerical_features].fillna(val[numerical_features].median(), inplace=False)
    test[numerical_features] = test[numerical_features].fillna(test[numerical_features].median(), inplace=False)

    # for categorical feature filling by majority
    train[nominal_features] = train[nominal_features].fillna(
        train[nominal_features].agg(lambda x: x.value_counts().index[0]),
        inplace=False)
    val[nominal_features] = val[nominal_features].fillna(val[nominal_features].agg(lambda x: x.value_counts().index[0]),
                                                         inplace=False)
    test[nominal_features] = test[nominal_features].fillna(
        test[nominal_features].agg(lambda x: x.value_counts().index[0]), inplace=False)

    train = train.drop(label, axis=1)
    val = val.drop(label, axis=1)
    test = test.drop(label, axis=1)

    return train, val, test


def closest_fit(ref_data, examine_row, local_nominal_features,
                local_numerical_features):
    current_nominal_features = [f for f in local_nominal_features if
                                f in ref_data.columns]
    data_nominal = ref_data[current_nominal_features]
    examine_row_obj = examine_row[current_nominal_features].values
    obj_diff = data_nominal.apply(
        lambda _row: (_row.values != examine_row_obj).sum(), axis=1)

    num_features = [f for f in local_numerical_features if
                    f in ref_data.columns]
    data_numerical = ref_data[num_features]
    examine_row_numerical = examine_row[num_features]
    col_max = data_numerical.max().values
    col_min = data_numerical.min().values
    r = col_max - col_min

    # replace missing values in examine row to inf in order distance to work
    examine_row_numerical = examine_row_numerical.replace(np.nan, np.inf)

    num_diff = data_numerical.apply(
        lambda _row: __distance_num(_row.values, examine_row_numerical.values,
                                    r), axis=1)
    for row in num_diff:
        row[(row == np.inf)] = 1

    num_diff = num_diff.apply(lambda _row: _row.sum())

    total_dist = num_diff + obj_diff
    return total_dist.reset_index(drop=True).idxmin()


def __distance_num(a, b, r):
    np.seterr(invalid='ignore')
    return np.divide(np.abs(np.subtract(a, b)), r)


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


def change_negative_to_null(x_train: DataFrame, x_val: DataFrame, x_test: DataFrame) -> (
DataFrame, DataFrame, DataFrame):

    x_train[x_train < 0] = np.nan
    x_val[x_val < 0] = np.nan
    x_test[x_test < 0] = np.nan

    return x_train, x_val, x_test


if __name__ == '__main__':
    main()
