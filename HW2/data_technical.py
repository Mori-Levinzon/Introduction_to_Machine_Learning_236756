from os import path
from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# dir
PATH = path.dirname(path.realpath(__file__)) + "/"
DATA_FILENAME = "ElectionsData.csv"
DATA_PATH = PATH + DATA_FILENAME
SELECTED_FEATURES_PATH = PATH + "rawSelectedFeatures.csv"

# constants
global_train_size = 0.60
global_validation_size = 0.20
global_test_size = 1 - global_train_size - global_validation_size
global_z_threshold = 4.5
global_correlation_threshold = 0.9
global_variance_threshold = 0.2
label = 'Vote'

total_features = ['Vote', 'Occupation_Satisfaction',
                'Avg_monthly_expense_when_under_age_21',
                'Avg_lottary_expanses',
                'Most_Important_Issue', 'Avg_Satisfaction_with_previous_vote',
                'Looking_at_poles_results',
                'Garden_sqr_meter_per_person_in_residancy_area', 'Married',
                'Gender',
                'Voting_Time', 'Financial_balance_score_(0-1)',
                '%Of_Household_Income',
                'Avg_government_satisfaction', 'Avg_education_importance',
                'Avg_environmental_importance', 'Avg_Residancy_Altitude',
                'Yearly_ExpensesK', '%Time_invested_in_work', 'Yearly_IncomeK',
                'Avg_monthly_expense_on_pets_or_plants',
                'Avg_monthly_household_cost',
                'Will_vote_only_large_party', 'Phone_minutes_10_years',
                'Avg_size_per_room', 'Weighted_education_rank',
                '%_satisfaction_financial_policy',
                'Avg_monthly_income_all_years',
                'Last_school_grades', 'Age_group',
                'Number_of_differnt_parties_voted_for',
                'Political_interest_Total_Score',
                'Number_of_valued_Kneset_members',
                'Main_transportation', 'Occupation', 'Overall_happiness_score',
                'Num_of_kids_born_last_10_years', 'Financial_agenda_matters']

labeless_features = ['Occupation_Satisfaction',
                          'Avg_monthly_expense_when_under_age_21',
                          'Avg_lottary_expanses',
                          'Most_Important_Issue',
                          'Avg_Satisfaction_with_previous_vote',
                          'Looking_at_poles_results',
                          'Garden_sqr_meter_per_person_in_residancy_area',
                          'Married', 'Gender',
                          'Voting_Time', 'Financial_balance_score_(0-1)',
                          '%Of_Household_Income',
                          'Avg_government_satisfaction',
                          'Avg_education_importance',
                          'Avg_environmental_importance',
                          'Avg_Residancy_Altitude',
                          'Yearly_ExpensesK', '%Time_invested_in_work',
                          'Yearly_IncomeK',
                          'Avg_monthly_expense_on_pets_or_plants',
                          'Avg_monthly_household_cost',
                          'Will_vote_only_large_party',
                          'Phone_minutes_10_years',
                          'Avg_size_per_room', 'Weighted_education_rank',
                          '%_satisfaction_financial_policy',
                          'Avg_monthly_income_all_years',
                          'Last_school_grades', 'Age_group',
                          'Number_of_differnt_parties_voted_for',
                          'Political_interest_Total_Score',
                          'Number_of_valued_Kneset_members',
                          'Main_transportation', 'Occupation',
                          'Overall_happiness_score',
                          'Num_of_kids_born_last_10_years',
                          'Financial_agenda_matters']

nominal_features = ['Most_Important_Issue', 'Looking_at_poles_results',
                    'Married',
                    'Gender', 'Voting_Time', 'Will_vote_only_large_party',
                    'Age_group',
                    'Main_transportation', 'Occupation',
                    'Financial_agenda_matters']

numerical_features = ['Occupation_Satisfaction',
                      'Avg_monthly_expense_when_under_age_21',
                      'Avg_lottary_expanses',
                      'Avg_Satisfaction_with_previous_vote',
                      'Garden_sqr_meter_per_person_in_residancy_area',
                      'Financial_balance_score_(0-1)', '%Of_Household_Income',
                      'Avg_government_satisfaction',
                      'Avg_education_importance',
                      'Avg_environmental_importance', 'Avg_Residancy_Altitude',
                      'Yearly_ExpensesK', '%Time_invested_in_work',
                      'Yearly_IncomeK',
                      'Avg_monthly_expense_on_pets_or_plants',
                      'Avg_monthly_household_cost', 'Phone_minutes_10_years',
                      'Avg_size_per_room', 'Weighted_education_rank',
                      '%_satisfaction_financial_policy',
                      'Avg_monthly_income_all_years', 'Last_school_grades',
                      'Number_of_differnt_parties_voted_for',
                      'Political_interest_Total_Score',
                      'Number_of_valued_Kneset_members',
                      'Overall_happiness_score',
                      'Num_of_kids_born_last_10_years']

multi_nominal_features = ['Vote', 'Most_Important_Issue',
                          'Main_transportation', 'Occupation']

uniform_features = ['Occupation_Satisfaction', 'Looking_at_poles_results',
                    'Married', 'Gender',
                    'Voting_Time', 'Financial_balance_score_(0-1)',
                    '%Of_Household_Income',
                    'Avg_government_satisfaction', 'Avg_education_importance',
                    'Avg_environmental_importance',
                    'Avg_Residancy_Altitude', 'Yearly_ExpensesK',
                    '%Time_invested_in_work',
                    '%_satisfaction_financial_policy', 'Age_group',
                    'Main_transportation', 'Occupation',
                    'Financial_agenda_matters']

normal_features = ['Garden_sqr_meter_per_person_in_residancy_area',
                   'Yearly_IncomeK',
                   'Avg_monthly_expense_on_pets_or_plants',
                   'Avg_monthly_household_cost', 'Avg_size_per_room',
                   'Number_of_differnt_parties_voted_for',
                   'Political_interest_Total_Score',
                   'Number_of_valued_Kneset_members',
                   'Overall_happiness_score']


def export_to_csv(filespath: str, x_train: DataFrame, x_val: DataFrame,
                  x_test: DataFrame, y_train: DataFrame, y_val: DataFrame,
                  y_test: DataFrame, prefix: str):
    x_train = x_train.assign(Vote=y_train.values)
    x_val = x_val.assign(Vote=y_val.values)
    x_test = x_test.assign(Vote=y_test.values)
    x_train.to_csv(filespath + "{}_train.csv".format(prefix), index=False)
    x_val.to_csv(filespath + "{}_val.csv".format(prefix), index=False)
    x_test.to_csv(filespath + "{}_test.csv".format(prefix), index=False)


def split_database(df: DataFrame, test_size: float, validation_size: float):
    validation_after_split_size = validation_size / (1 - test_size)
    x_train, x_test, y_train, y_test = train_test_split(
        df.loc[:, df.columns != label], df[label],
        test_size=test_size,
        shuffle=True, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=validation_after_split_size,
                                                      shuffle=True,
                                                      random_state=0)
    return x_train, x_val, x_test, y_train, y_val, y_test


def categorize_data(df: DataFrame):
    object_columns = df.keys()[df.dtypes.map(lambda x: x == 'object')]#get the data columns
    for curr_column in object_columns:#iterate over columns
        df[curr_column] = df[curr_column].astype("category")
        # change categories to int values
        # should have ".astype('int32')"
        df[curr_column + '_Int'] = df[curr_column].cat.rename_categories(range(df[curr_column].nunique()))
        df.loc[df[curr_column].isna(), curr_column + '_Int'] = np.nan  # fix NaN conversion
        df[curr_column] = df[curr_column + '_Int']
        '_Int'
        df = df.drop(curr_column + '_Int', axis=1)
    return df


def load_data(filepath: str) -> DataFrame:
    df = pd.read_csv(filepath, header=0)
    return df

def heat_map(df: DataFrame):
    plt.close('all')
    plt.subplots(figsize=(28, 32))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, square=False, annot=True, fmt='.1f', vmax=1.0, vmin=-1.0, cmap="RdBu", linewidths=2).set_title(
        'Correlation Matrix')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.savefig("graphic_analysis/heatmap.png")
    plt.show()


def features_histograms(df: DataFrame):
    plt.close('all')
    local_all_features = list(df.keys())
    for f in local_all_features:
        plt.title(f)
        plt.hist(df[f].values)
        plt.savefig("histograms/{}.png".format(f))
        plt.show()


def data_information():
    df = load_data(DATA_PATH)
    # categorized nominal attributes to int
    df = categorize_data(df)
    # features_histograms(df)
    heat_map(df)

def feature_label_relationship():
    df = load_data(DATA_PATH)
    # categorized nominal attributes to int
    df = categorize_data(df)
    y_df = df[label]
    x_df = df.drop(label, axis=1)
    plt.close('all')
    local_all_features = sorted(list(x_df.keys()))
    print("label map: {}".format({'Blues': 0, 'Browns': 1, 'Greens': 2, 'Greys': 3, 'Khakis': 4, 'Oranges': 5, 'Pinks': 6,
                                  'Purples': 7, 'Reds': 8, 'Turquoises': 9, 'Violets': 10, 'Whites': 11, 'Yellows': 12}))
    for f in local_all_features:
        print(f)
        plt.scatter(x_df[f], y_df)
        plt.ylabel(label, color='b')
        plt.xlabel(f)
        plt.savefig("graphic_analysis/{}.png".format(label + "_Vs._" + f))
        plt.show()


if __name__ == '__main__':
    # data_information()
    feature_label_relationship()
