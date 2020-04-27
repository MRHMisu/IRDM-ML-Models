import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(input_data_path):
    p = Path(input_data_path)
    with p.open('rb') as f:
        X = np.load(f)
    print(X.shape)
    Y = X[:, -1]
    X = X[:, :-1]
    print(X.shape)
    print(Y.shape)
    count_classes = pd.value_counts(Y, sort=True)
    print(count_classes)


if __name__ == "__main__":
    sample_training = "../../dataset/fv_train_sample.npy"
    sample_validation = "../../dataset/fv_validation_sample.npy"
    load_data(sample_training)
    load_data(sample_validation)
    sys.exit();
