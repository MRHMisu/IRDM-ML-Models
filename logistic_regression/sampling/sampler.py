import sys
from collections import Counter
from pathlib import Path

import numpy as np
from imblearn.under_sampling import NearMiss


def load_data(input_data_path):
    'loading the file from specified path'

    p = Path(input_data_path)
    with p.open('rb') as f:
        X = np.load(f)
    Y = X[:, -1]
    X = X[:, :-1]
    return X, Y


def under_sampling(input_data_path, out_sample_path):
    'Perform NearMiss under sampling to balance the dataset'

    X, Y = load_data(input_data_path)
    near_miss = NearMiss()
    X_res, y_res = near_miss.fit_sample(X, Y)
    print('Original data shape {}'.format(Counter(Y)))
    print('Resampled data shape {}'.format(Counter(y_res)))
    resampled_validation = np.c_[X_res, y_res]
    np.save(out_sample_path, resampled_validation);


if __name__ == "__main__":
    sample_training = "../../dataset/fv_train_sample.npy"
    sample_validation = "../../dataset/fv_validation_sample.npy"

    train__fv = "../dataset/fv_train.npy"
    validate_fv = "../dataset/fv_validation.npy"

    under_sampling(train__fv, sample_training)
    under_sampling(validate_fv, sample_validation)

    sys.exit();
