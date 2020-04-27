import sys
from pathlib import Path

import numpy as np
from sklearn import metrics


def load_data(input_data_path):
    p = Path(input_data_path)
    with p.open('rb') as f:
        X = np.load(f)
    Y = X[:, -1]
    X = X[:, :-1]
    return X, Y


def normalize_features(features):
    min_values = np.min(features, axis=0)
    max_values = np.max(features, axis=0)
    ranges = max_values - min_values
    normalized_features = 1 - ((max_values - features) / ranges)
    return normalized_features


def sigmoid(coefficients, feature_vectors):
    return 1.0 / (1 + np.exp(-np.dot(feature_vectors, coefficients.T)))


def gradient_function(coefficients, features, pre_label):
    first_derivative = sigmoid(coefficients, features) - pre_label.reshape(
        features.shape[0], -1)
    final_derivative = np.dot(first_derivative.T, features)
    return final_derivative


def cost_function(coefficients, features, pre_label):
    log_likelihood = sigmoid(coefficients, features)
    pre_label = np.squeeze(pre_label)
    step1 = pre_label * np.log(log_likelihood)
    step2 = (1 - pre_label) * np.log(1 - log_likelihood)
    final = -step1 - step2
    return np.mean(final)


def calculate_gradient_descent(features, pre_label, coefficients, learning_rate, number_of_iteration):
    current_cost = cost_function(coefficients, features, pre_label)
    training_loss = []
    for index in range(number_of_iteration):
        previous_cost = current_cost
        coefficients = coefficients - (learning_rate * gradient_function(coefficients, features, pre_label))
        current_cost = cost_function(coefficients, features, pre_label)
        loss = previous_cost - current_cost
        training_loss.append(loss)
    return coefficients, training_loss


def predict(features, coefficients):
    predicted_probability = sigmoid(coefficients, features)
    predicted_class = np.where(predicted_probability >= .5, 1, 0)
    return np.squeeze(predicted_class)


if __name__ == "__main__":
    sample_training = "../dataset/fv_train_sample.npy"
    sample_validation = "../dataset/fv_validation_sample.npy"
    coefficients_out = "../result/logistic-coefficients.txt"

    features, pre_label = load_data(sample_training)
    features = normalize_features(features)
    features = np.hstack((np.asmatrix(np.ones(features.shape[0])).T, features))

    learning_rate = 0.001
    number_of_iteration = 10000

    coefficients = np.asmatrix(np.zeros(features.shape[1]))
    coefficients, training_loss = calculate_gradient_descent(features, pre_label, coefficients, learning_rate,
                                                             number_of_iteration)

    np.savetxt(coefficients_out, coefficients, delimiter=",")

    features_val, pre_label_val = load_data(sample_validation)
    features_val = normalize_features(features_val)
    features_val = np.hstack((np.asmatrix(np.ones(features_val.shape[0])).T, features_val))

    predicted_label = predict(features_val, coefficients)
    cnf_matrix = metrics.confusion_matrix(pre_label_val, predicted_label)
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(pre_label_val, predicted_label))
    print("Precision:", metrics.precision_score(pre_label_val, predicted_label, zero_division='warn'))
    print("Recall:", metrics.recall_score(pre_label_val, predicted_label, zero_division='warn'))

    sys.exit()
