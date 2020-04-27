import sys
from pathlib import Path

import numpy as np
import xgboost as xgb


def get_dmatrix(path):
    p = Path(path)
    with p.open('rb') as f:
        X = np.load(f)
    Y = X[:, -1]
    X = X[:, :-1]
    return xgb.DMatrix(X, label=Y)


def get_cv(params, num_boost_round, dtrain):
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=40,
        nfold=3,
        metrics={'ndcg'},
        early_stopping_rounds=10
    )
    return cv_results


def tune_number_of_iteration(params, num_boost_round, dtrain, dvalidation):
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalidation, "Test")],
        early_stopping_rounds=10
    )
    best_score = model.best_score
    best_iteration = model.best_iteration + 1;
    return best_score, best_iteration


def tuning_max_depth_child_weight(params, num_boost_round, dtrain):
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(3, 9)
        for min_child_weight in range(3, 9)
    ]
    min_ndcg = 0.0
    best_max_depth = None
    best_min_child_weight = None
    for max_depth, min_child_weight in gridsearch_params:
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        cv_results = get_cv(params, num_boost_round, dtrain)
        mean_ndcg = cv_results['test-ndcg-mean'].max()
        boost_rounds = cv_results['test-ndcg-mean'].argmin()
        if mean_ndcg > min_ndcg:
            min_ndcg = mean_ndcg
            best_max_depth = max_depth
            best_min_child_weight = min_child_weight
    return best_max_depth, best_min_child_weight, boost_rounds


def tuning_subsample_colsample_bytree(params, num_boost_round, dtrain):
    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i / 10. for i in range(3, 9)]
        for colsample in [i / 10. for i in range(3, 9)]
    ]
    min_ndcg = 0.0
    best_subsample = None
    best_colsample = None

    for subsample, colsample in reversed(gridsearch_params):
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample
        cv_results = get_cv(params, num_boost_round, dtrain)
        mean_ndcg = cv_results['test-ndcg-mean'].max()
        boost_rounds = cv_results['test-ndcg-mean'].argmin()
        if mean_ndcg > min_ndcg:
            min_ndcg = mean_ndcg
            best_subsample = subsample
            best_colsample = colsample
    return best_subsample, best_colsample, boost_rounds


def tuning_eta(params, num_boost_round, dtrain):
    min_ndcg = 0.0
    best_eta = None
    for eta in [.005, .01, .05, .1, .2, .3]:
        params['eta'] = eta
        cv_results = get_cv(params, num_boost_round, dtrain)
        mean_ndcg = cv_results['test-ndcg-mean'].max()
        boost_rounds = cv_results['test-ndcg-mean'].argmin()
        if mean_ndcg > min_ndcg:
            min_ndcg = mean_ndcg
            best_eta = eta
    return best_eta, boost_rounds


if __name__ == "__main__":
    sample_validation = "../dataset/fv_validation_sample.npy"
    validate_fv = "../dataset/fv_validation.npy"

    dtrain = get_dmatrix(validate_fv)
    dval = get_dmatrix(validate_fv)

    num_boost_round = 100

    # base  params initialization
    params = {
        'max_depth': 2,
        'min_child_weight': 1,
        'eta': .3,
        'subsample': 1,
        'colsample_bytree': 1,
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg'
    }

    print(params)

    best_score, best_iteration = tune_number_of_iteration(params, num_boost_round, dtrain, dtrain)
    best_max_depth, best_min_child_weight = tuning_max_depth_child_weight(params, num_boost_round, dtrain)
    best_subsample, best_colsample = tuning_subsample_colsample_bytree(params, num_boost_round, dtrain)
    best_eta = tuning_eta(params, num_boost_round, dtrain)

    print(best_iteration)
    params['max_depth'] = best_max_depth
    params['min_child_weight'] = best_min_child_weight
    params['subsample'] = best_subsample
    params['colsample_bytree'] = best_colsample
    params['eta'] = best_eta

    print(params)
    sys.exit()
