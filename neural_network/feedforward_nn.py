import numpy as np
from sklearn import metrics
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.saving.save import load_model


def load_data(train_path):
    'loading the data form the specified path'

    train = np.load(train_path)
    y = train[:, -1]
    x = train[:, :-1]
    return x, y


def build_feedforward_nn(X, y, save_path):
    'building the proposed NN architecture'

    model = Sequential()
    model.add(Dense(15, input_dim=100, activation='relu'))  # input layer
    model.add(Dense(10, activation='relu'))  # hidden layer I
    model.add(Dense(5, activation='relu'))  # hidden layer II
    model.add(Dense(1, activation='sigmoid'))  # output layer
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=10)
    model.save(save_path)
    return model;


def evaluate(model, X_v, y_v):
    'evaluating model performance'

    loss, accuracy = model.evaluate(X_v, y_v)
    print('Loss: %.2f' % (loss))
    print('Accuracy: %.2f' % (accuracy * 100))


if __name__ == "__main__":
    sample_training = "../dataset/fv_train_sample.npy"
    sample_validation = "../dataset/fv_validation_sample.npy"
    save_path = "../result/nn_model.h5"

    X, y = load_data(sample_training)
    X_v, y_v = load_data(sample_validation)

    # model = build_feedforward_nn(X, y, save_path)
    model = load_model(save_path)
    evaluate(model, X_v, y_v)
    pre_y = model.predict(X_v)
    pre_y = np.where(pre_y >= .5, 1, 0)
    cnf_matrix = metrics.confusion_matrix(y_v, pre_y)
    print("Confusion Matrix:\n")
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_v, pre_y))
    print("Precision:", metrics.precision_score(y_v, pre_y, zero_division='warn'))
    print("Recall:", metrics.recall_score(y_v, pre_y, zero_division='warn'))
    exit()
