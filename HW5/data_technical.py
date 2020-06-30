from os import path
from pandas import DataFrame, Series, read_csv, concat
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
import operator
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


PATH = path.dirname(path.realpath(__file__)) + "/"

TRAINING_SET_PATH = PATH + "ElectionsData.csv"
TEST_SET_PATH = PATH + "ElectionsData_Pred_Features.csv"

TRAIN_PATH = PATH + "fixed_labeled_train.csv"
VALIDATION_PATH = PATH + "fixed_labeled_val.csv"
TEST_PATH = PATH + "fixed_labeled_test.csv"
TEST_UNLABELED_PATH = PATH + "fixed_unlabeled_test.csv"
EXPORT_TEST_PREDICTIONS = PATH + "test_predictions.csv"

# constants
global_train_size = 0.65
global_test_size = 0.25
global_validation_size = 0.1
assert global_train_size + global_test_size + global_validation_size == 1

global_z_threshold = 4.5
global_correlation_threshold = 0.9
label = 'Vote'
global_party_in_coalition_threshold = 0.95
voters_id = "IdentityCard_Num"
# lists

selected_features = ['Vote', 'Yearly_IncomeK', 'Avg_size_per_room',
                     'Avg_Satisfaction_with_previous_vote', 'Weighted_education_rank',
                     'Avg_monthly_income_all_years', 'Most_Important_Issue',
                     'Number_of_differnt_parties_voted_for', 'Political_interest_Total_Score',
                     'Overall_happiness_score']

selected_features_without_label = ['Yearly_IncomeK', 'Avg_size_per_room',
                                   'Avg_Satisfaction_with_previous_vote', 'Weighted_education_rank',
                                   'Avg_monthly_income_all_years', 'Most_Important_Issue',
                                   'Number_of_differnt_parties_voted_for', 'Political_interest_Total_Score',
                                   'Overall_happiness_score']

selected_nominal_features = ['Most_Important_Issue']

selected_numerical_features = ['Yearly_IncomeK', 'Avg_size_per_room',
                               'Avg_Satisfaction_with_previous_vote', 'Weighted_education_rank',
                               'Avg_monthly_income_all_years',
                               'Number_of_differnt_parties_voted_for', 'Political_interest_Total_Score',
                               'Overall_happiness_score']

selected_multi_nominal_features = ['Most_Important_Issue']

selected_uniform_features = ['Yearly_IncomeK']

selected_normal_features = ['Avg_size_per_room',
                            'Avg_Satisfaction_with_previous_vote', 'Weighted_education_rank',
                            'Avg_monthly_income_all_years',
                            'Number_of_differnt_parties_voted_for', 'Political_interest_Total_Score',
                            'Overall_happiness_score']

label2num = {'Blues': 0, 'Browns': 1, 'Greens': 2, 'Greys': 3, 'Khakis': 4, 'Oranges': 5, 'Pinks': 6, 'Purples': 7,
             'Reds': 8,
             'Turquoises': 9, 'Violets': 10, 'Whites': 11, 'Yellows': 12}

BrownBinaryTree2num = {'Not_Browns': 0, 'Browns': 1}

num2label = {0: 'Blues', 1: 'Browns', 2: 'Greens', 3: 'Greys', 4: 'Khakis', 5: 'Oranges', 6: 'Pinks', 7: 'Purples',
             8: 'Reds',
             9: 'Turquoises', 10: 'Violets', 11: 'Whites', 12: 'Yellows'}


def import_from_csv(filepath: str) -> DataFrame:
    df = read_csv(filepath, header=0)
    return df


def export_to_csv(filepath: str, df: DataFrame):
    df.to_csv(filepath, index=False)


def filter_possible_coalitions(possible_coalitions: dict):
    """
    :param possible_coalitions: all possible coalition
    :return: possible coalition without duplication
    """
    # remove duplicates
    filtered_possible_coalitions = dict()
    for _coalition_name, _coalition_list in possible_coalitions.items():
        _coalition_list.sort()
        if _coalition_list not in filtered_possible_coalitions.values():
            filtered_possible_coalitions[_coalition_name] = _coalition_list
    return filtered_possible_coalitions


def to_binary_class(data, value):
    """
    :param data: regular data
    :param value: the value to be assigned as 1
    :return: binary classified data
    """
    binary_data = data.copy()
    bool_labels = binary_data[label] == value
    binary_data[label] = bool_labels
    return binary_data


def divide_data(df: DataFrame, data_class=label):
    x_df = df.loc[:, df.columns != data_class]
    y_df = df[data_class]
    return x_df, y_df


def categorize_data(df: DataFrame):
    object_columns = df.keys()[df.dtypes.map(lambda x: x == 'object')]
    for curr_column in object_columns:
        df[curr_column] = df[curr_column].astype("category")
        df[curr_column + '_Int'] = df[curr_column].cat.rename_categories(range(df[curr_column].nunique()))
        df.loc[df[curr_column].isna(), curr_column + '_Int'] = np.nan  # fix NaN conversion
        df[curr_column] = df[curr_column + '_Int']
        df = df.drop(curr_column + '_Int', axis=1)
    return df


def split_database(df: DataFrame, test_size: float, validation_size: float) -> (
        DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame):
    validation_after_split_size = validation_size / (1 - test_size)
    first_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    x = df.loc[:, df.columns != label]
    y = df[label]
    train_index_first, test_index = next(first_split.split(x, y))
    x_train, x_test, y_train, y_test = x.iloc[train_index_first], x.iloc[test_index], y[train_index_first], y[test_index]

    test = x_test.assign(Vote=y_test.values).reset_index(drop=True)
    train = x_train.assign(Vote=y_train.values).reset_index(drop=True)

    x = train.loc[:, df.columns != label]
    y = train[label]

    second_split = StratifiedShuffleSplit(n_splits=1, test_size=validation_after_split_size)
    train_index_second, val_index = next(second_split.split(x, y))
    x_train, x_val, y_train, y_val = x.iloc[train_index_second], x.iloc[val_index], y[train_index_second], y[val_index]

    train = x_train.assign(Vote=y_train.values).reset_index(drop=True)
    val = x_val.assign(Vote=y_val.values).reset_index(drop=True)

    return train, val, test


def score(x_train: DataFrame, y_train: DataFrame, clf, k: int):
    return cross_val_score(clf, x_train, y_train, cv=k, scoring='accuracy').mean()


def get_sorted_vote_division(y):
    vote_results = dict()
    for label_name, label_index in label2num.items():
        percent_of_voters = sum(list(y == label_index)) / len(y)
        vote_results[label_index] = percent_of_voters
    return OrderedDict(sorted(vote_results.items(), key=operator.itemgetter(1)))


def export_results_to_csv(predictions_vector: np.ndarray, voters_id_col):
    """
    Straight forward from its name
    :param voters_id_col: voters id column
    :param predictions_vector: The predicted Series of Votes
    :return: None
    """
    d = {voters_id: voters_id_col, "PredictVote": predictions_vector}
    DataFrame(d).to_csv(EXPORT_TEST_PREDICTIONS, index=False)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
