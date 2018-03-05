"""
keras mlp sample
"""
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras import metrics
from keras.utils import np_utils
import numpy as np
from sklearn import datasets


def main():
    # load dataset
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    output_size = np.unique(y).shape[0]
    y = np_utils.to_categorical(y)

    # build model
    model = Sequential()
    model.add(Dense(units=10, activation='relu', input_dim=x.shape[-1]))
    model.add(Dropout(0.2))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_size, activation='softmax'))

    # fit model
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01), metrics=[metrics.categorical_accuracy, metrics.mse])
    model.fit(x, y, epochs=100, batch_size=10)

    # evaluate model
    results = model.evaluate(x, y)
    print(f"Loss    : {results[0]}")
    print(f"Accuracy: {results[1]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
