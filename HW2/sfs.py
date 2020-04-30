from HW2.data_prepration import *
from HW2.data_technical import *
import warnings

warnings.filterwarnings('ignore')


def sfs_algo(x_train: DataFrame, y_train: DataFrame, clf, subset_size: int = None):
    """
    :param x_train: DataFrame
    :param y_train: labels
    :param clf: classifier to examine
    :param subset_size: user required subset size
    :return: selected feature subset
    """
    subset_selected_features = []
    best_total_score = float('-inf')

    if subset_size:
        subset_size = min(len(labeless_features), subset_size)
    else:
        subset_size = len(labeless_features)

    for _ in range(subset_size):
        best_score = float('-inf')
        best_feature = None
        unselect_features = [f for f in labeless_features if f not in subset_selected_features]
        for f in unselect_features:
            current_features = subset_selected_features + [f]
            current_score = score(x_train[current_features], y_train, clf)
            if current_score > best_score:
                best_score = current_score
                best_feature = f
        if best_score > best_total_score:
            best_total_score = best_score
            subset_selected_features.append(best_feature)
        else:
            break
    return subset_selected_features


def run_sfs_base_clfs(x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame, x_val: DataFrame, y_val: DataFrame):
    # examine sfs algorithm with SVM
    dtc = SGDClassifier(random_state=92, max_iter=1000, tol=1e-3)
    score_before_sfs = score(x_train, y_train, dtc)
    print("SVM Classifier accuracy score before SFS is: {}".format(score_before_sfs))

    selected_features_dtc = sfs_algo(x_train, y_train, dtc)
    print("SVM Classifier selected features are: {}".format(selected_features_dtc))

    score_after_sfs = score(x_train[selected_features_dtc], y_train, dtc)
    print("SVM Classifier score after SFS is: {}".format(score_after_sfs))

    # examine sfs algorithm with K Neighbors Classifier

    knn = KNeighborsClassifier(n_neighbors=5)
    score_before_sfs = score(x_train, y_train, knn)
    print("K Neighbors Classifier score before SFS is: {}".format(score_before_sfs))

    selected_features_knn = sfs_algo(x_train, y_train, knn)
    print("K Neighbors Classifier selected features are: {}".format(selected_features_knn))

    score_after_sfs = score(x_train[selected_features_knn], y_train, knn)
    print("K Neighbors Classifier score after SFS is: {}".format(score_after_sfs))

    return selected_features_dtc, selected_features_knn