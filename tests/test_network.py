"""Integration tests for the Sequential model."""

import numpy as np
import pytest

from nn import Sequential, Dense, ReLU, Sigmoid, Softmax, MSE, CrossEntropy, BinaryCrossEntropy, Adam, SGD
from nn.layers import ActivationLayer


def make_regression_data(n=200, seed=0):
    np.random.seed(seed)
    X = np.random.randn(n, 2)
    y = (X[:, 0] + X[:, 1]).reshape(-1, 1)
    return X, y


def make_classification_data(n=200, seed=0):
    np.random.seed(seed)
    X = np.random.randn(n, 2)
    labels = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, labels


def test_sequential_compile_required():
    model = Sequential([Dense(2, 1)])
    with pytest.raises(RuntimeError):
        model.fit(np.zeros((5, 2)), np.zeros((5, 1)), epochs=1, verbose=False)


def test_sequential_predict_shape():
    model = Sequential([Dense(2, 4), ActivationLayer(ReLU()), Dense(4, 1)])
    model.compile(loss=MSE(), optimizer=Adam())
    X, y = make_regression_data()
    model.fit(X, y, epochs=2, verbose=False)
    preds = model.predict(X)
    assert preds.shape == (200, 1)


def test_sequential_regression_loss_decreases():
    np.random.seed(1)
    X, y = make_regression_data()
    model = Sequential([Dense(2, 8), ActivationLayer(ReLU()), Dense(8, 1)])
    model.compile(loss=MSE(), optimizer=Adam(lr=0.01))
    history = model.fit(X, y, epochs=50, verbose=False)
    assert history["loss"][-1] < history["loss"][0]


def test_sequential_binary_classification():
    X, labels = make_classification_data()
    y = labels.reshape(-1, 1).astype(float)
    model = Sequential([Dense(2, 8), ActivationLayer(ReLU()), Dense(8, 1), ActivationLayer(Sigmoid())])
    model.compile(loss=BinaryCrossEntropy(), optimizer=Adam(lr=0.01))
    history = model.fit(X, y, epochs=100, verbose=False)
    assert history["loss"][-1] < history["loss"][0]


def test_sequential_multiclass_classification():
    np.random.seed(42)
    X = np.random.randn(300, 4)
    labels = np.argmax(X[:, :3], axis=1)
    model = Sequential([
        Dense(4, 16),
        ActivationLayer(ReLU()),
        Dense(16, 3),
        ActivationLayer(Softmax()),
    ])
    model.compile(loss=CrossEntropy(), optimizer=Adam(lr=0.01))
    history = model.fit(X, labels, epochs=50, verbose=False)
    assert history["loss"][-1] < history["loss"][0]


def test_sequential_validation_data():
    X, y = make_regression_data()
    X_val, y_val = make_regression_data(n=50, seed=99)
    model = Sequential([Dense(2, 4), ActivationLayer(ReLU()), Dense(4, 1)])
    model.compile(loss=MSE(), optimizer=SGD(lr=0.01))
    history = model.fit(X, y, epochs=10, verbose=False, validation_data=(X_val, y_val))
    assert "val_loss" in history
    assert len(history["val_loss"]) == 10


def test_sequential_evaluate():
    X, y = make_regression_data()
    model = Sequential([Dense(2, 1)])
    model.compile(loss=MSE(), optimizer=Adam())
    model.fit(X, y, epochs=1, verbose=False)
    loss = model.evaluate(X, y)
    assert loss >= 0


def test_sequential_add():
    model = Sequential()
    model.add(Dense(2, 4)).add(ActivationLayer(ReLU())).add(Dense(4, 1))
    assert len(model.layers) == 3
