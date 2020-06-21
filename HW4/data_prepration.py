from HW4.data_technical import *
from impyute import imputation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


def main():
    df = load_data(DATA_PATH)

    # categorized nominal attributes to int
    df = categorize_data(df)

    # export raw data to csv files
    train, val, test = split_database(df, global_test_size, global_validation_size)

    export_to_csv(PATH + "raw_train.csv", train)
    export_to_csv(PATH + "raw_val.csv", val)
    export_to_csv(PATH + "raw_test.csv", test)

    # data cleansing
    train, val, test = change_negative_to_null(train, val, test)
    train, val, test = remove_outlier_values(train, val, test, global_z_threshold)

    # imputation
    train, val, test = imputations(train, val, test)

    # selected features
    train = train[selected_features]
    test = test[selected_features]
    val = val[selected_features]

    # scaling
    train, val, test = normalization(train, val, test)

    export_to_csv(TRAIN_PATH, train)
    export_to_csv(VALIDATION_PATH, val)
    export_to_csv(TEST_PATH, test)


def change_negative_to_null(x_train: DataFrame, x_val: DataFrame,
                   x_test: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    for feature in selected_numerical_features:
        x_train.loc[(~x_train[feature].isnull()) & (
                x_train[feature] < 0), feature] = np.nan
        x_val.loc[(~x_val[feature].isnull()) & (
                x_val[feature] < 0), feature] = np.nan
        x_test.loc[(~x_test[feature].isnull()) & (
                x_test[feature] < 0), feature] = np.nan
    return x_train, x_val, x_test


def remove_outlier_values(x_train: DataFrame, x_val: DataFrame, x_test: DataFrame,
                    z_threshold: float):
    mean_train = x_train[selected_normal_features].mean()
    std_train = x_train[selected_normal_features].std()

    dist_train = (x_train[selected_normal_features] - mean_train) / std_train
    dist_val = (x_val[selected_normal_features] - mean_train) / std_train
    dist_test = (x_test[selected_normal_features] - mean_train) / std_train

    data_list = [x_train, x_val, x_test]
    dist_list = [dist_train, dist_val, dist_test]

    for feature in selected_normal_features:
        for df, dist in zip(data_list, dist_list):
            for i in dist[feature].loc[(dist[feature] > z_threshold) | (dist[feature] < -z_threshold)].index:
                df.at[i, feature] = np.nan

    return x_train, x_val, x_test


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
        row.fillna(ref_data.iloc[closest_fit(ref_data, row, selected_nominal_features, selected_numerical_features)], inplace=True)


def imputations(train: DataFrame, val: DataFrame, test: DataFrame):
    # fill missing values by using information from correlated features
    correlation_dict_train = get_features_correlation(train, global_correlation_threshold)

    fill_feature_correlation(train, correlation_dict_train)
    fill_feature_correlation(val, correlation_dict_train)
    fill_feature_correlation(test, correlation_dict_train)

    # fill missing data using closest fit
    print("closest fit for train")
    closest_fit_imputation(train.dropna(how='any'), train)
    print("closest fit for validation")
    closest_fit_imputation(val.dropna(how='any'), val)
    print("closest fit for test")
    closest_fit_imputation(test.dropna(how='any'), test)

    # fill normal distributed features using EM algorithm
    train_after_em = imputation.cs.em(np.array(train[selected_normal_features]), loops=50, dtype='cont')
    train.loc[:, selected_normal_features] = train_after_em

    test_after_em = imputation.cs.em(np.array(test[selected_normal_features]), loops=50, dtype='cont')
    test.loc[:, selected_normal_features] = test_after_em

    val_after_em = imputation.cs.em(np.array(val[selected_normal_features]), loops=50, dtype='cont')
    val.loc[:, selected_normal_features] = val_after_em

    # fill using statistics
    # for numerical feature filling by median
    train[selected_numerical_features] = train[selected_numerical_features].fillna(train[selected_numerical_features].median(),
                                                                                   inplace=False)
    val[selected_numerical_features] = val[selected_numerical_features].fillna(val[selected_numerical_features].median(), inplace=False)
    test[selected_numerical_features] = test[selected_numerical_features].fillna(test[selected_numerical_features].median(), inplace=False)

    # for categorical feature filling by majority
    train[selected_nominal_features] = train[selected_nominal_features].fillna(
        train[selected_nominal_features].agg(lambda x: x.value_counts().index[0]),
        inplace=False)
    val[selected_nominal_features] = val[selected_nominal_features].fillna(
        val[selected_nominal_features].agg(lambda x: x.value_counts().index[0]), inplace=False)
    test[selected_nominal_features] = test[selected_nominal_features].fillna(
        test[selected_nominal_features].agg(lambda x: x.value_counts().index[0]), inplace=False)

    return train, val, test


def normalization(x_train: DataFrame, x_val: DataFrame, x_test: DataFrame):
    scale_std = StandardScaler()
    scale_min_max = MinMaxScaler(feature_range=(-1, 1))
    local_uniform_features = [f for f in selected_uniform_features if f not in selected_nominal_features]
    x_train[local_uniform_features] = scale_min_max.fit_transform(x_train[local_uniform_features])
    x_val[local_uniform_features] = scale_min_max.transform(x_val[local_uniform_features])
    x_test[local_uniform_features] = scale_min_max.transform(x_test[local_uniform_features])

    local_non_uniform = [f for f in selected_features_without_label if
                         f not in selected_uniform_features and f not in selected_nominal_features]

    x_train[local_non_uniform] = scale_std.fit_transform(x_train[local_non_uniform])
    x_val[local_non_uniform] = scale_std.transform(x_val[local_non_uniform])
    x_test[local_non_uniform] = scale_std.transform(x_test[local_non_uniform])
    return x_train, x_val, x_test


if __name__ == '__main__':
    main()
