import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn import metrics


def get_DMatrix(path):
    p = Path(path)
    with p.open('rb') as f:
        X = np.load(f)
    Y = X[:, -1]
    X = X[:, :-1]

    return xgb.DMatrix(X, label=Y)


def save_best_model(params, num_boost_round, dtrain):
    best_model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")]
    )
    best_model.save_model("irdm.model")


if __name__ == "__main__":
    sample_training = "../dataset/fv_train_sample.npy"
    sample_validation = "../dataset/fv_validation_sample.npy"

    dtrain = get_DMatrix(sample_training)
    dtest = get_DMatrix(sample_validation)

    y_train = dtrain.get_label()
    y_test = dtest.get_label()

    # tuned parameters
    params = {'max_depth': 3, 'min_child_weight': 3, 'eta': 0.005, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'rank:ndcg', 'eval_metric': 'ndcg', 'validate_parameters': 1}

    num_boost_round = 98
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")]
    )

    model.save_model("../result/lamda_mart.model")

    y_pred = model.predict(dtest)
    predictions = np.where(y_pred >= .5, 1, 0)

    cnf_matrix = metrics.confusion_matrix(y_test, predictions)
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, predictions))
    print("Precision:", metrics.precision_score(y_test, predictions, zero_division='warn'))
    print("Recall:", metrics.recall_score(y_test, predictions, zero_division='warn'))
    sys.exit()
