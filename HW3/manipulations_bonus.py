from sklearn.ensemble import RandomForestClassifier
from HW3.data_technical import *
from pandas import concat, options
import numpy as np


class ManipulateData(object):
    def __init__(self):
        self.y_test: DataFrame = None
        self.y_train: DataFrame = None
        self.x_test: DataFrame = None
        self.x_train: DataFrame = None
        self.classifier_title = "RandomForest"
        self.classifier = RandomForestClassifier(random_state=0, criterion='gini', min_samples_split=3,
                                                 min_samples_leaf=1, n_estimators=500)

        self._current_manipulation = None

        self._manipulations_dict = \
            {
                "Avg_Satisfaction_with_previous_vote ": self.Avg_Satisfaction_with_previous_vote_increase,
                "Political_interest_Total_Score": self.Political_interest_Total_Score_increase,
                "Overall_happiness_score": self.Overall_happiness_score_decrease,
            }

    def Avg_Satisfaction_with_previous_vote_increase(self):
        options.mode.chained_assignment = None
        self.x_test.loc[self.x_test["Avg_Satisfaction_with_previous_vote"] <= 0.5, "Avg_Satisfaction_with_previous_vote"] = np.random.uniform(0.85, 1)

    def Political_interest_Total_Score_increase(self):
        options.mode.chained_assignment = None
        self.x_test.loc[self.x_test["Political_interest_Total_Score"] <= 0.5, "Political_interest_Total_Score"] = np.random.uniform(0.5, 0.7)

    def Overall_happiness_score_decrease(self):
        options.mode.chained_assignment = None
        self.x_test.loc[self.x_test["Overall_happiness_score"] >= -0.5, "Overall_happiness_score"] = np.random.uniform(-1, -0.8)

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.classifier.fit(x_train, y_train)

    def get_available_manipulations(self):
        return list(self._manipulations_dict.keys())

    def set_manipulation(self, manipulation: str):
        if manipulation not in self.get_available_manipulations():
            raise Exception("manipulation not available")
        self._current_manipulation = manipulation

    def predict(self, x_test, y_test):
        self.y_test = y_test
        self.x_test = x_test
        if self._current_manipulation is None:
            raise Exception("set manipulation first")
        self._manipulations_dict[self._current_manipulation]()
        print(f"current manipulation {self._current_manipulation}")
        winner_color(self.classifier, x_test)


def main():
    train_df = load_data(TRAIN_PATH)
    x_train = train_df[selected_features_without_label]
    y_train = train_df[label]

    validation_df = load_data(VALIDATION_PATH)
    x_val = validation_df[selected_features_without_label]
    y_val = validation_df[label]

    test_df = load_data(TEST_PATH)
    x_test = test_df[selected_features_without_label]
    y_test = test_df[label]

    manipulator = ManipulateData()
    manipulator.fit(concat([x_train, x_val], axis=0, join='outer', ignore_index=True),
                    concat([y_train, y_val], axis=0, join='outer', ignore_index=True))

    for man in manipulator.get_available_manipulations():
        manipulator.set_manipulation(man)
        manipulator.predict(x_test, y_test)


if __name__ == '__main__':
    main()
