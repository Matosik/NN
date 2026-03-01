"""Tests for loss functions."""

import numpy as np
import pytest

from nn.losses import MSE, CrossEntropy, BinaryCrossEntropy


def test_mse_zero_loss():
    mse = MSE()
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    loss = mse(y, y)
    assert loss == pytest.approx(0.0)


def test_mse_known_value():
    mse = MSE()
    y_pred = np.array([[2.0]])
    y_true = np.array([[0.0]])
    loss = mse(y_pred, y_true)
    assert loss == pytest.approx(4.0)


def test_mse_backward_shape():
    mse = MSE()
    y_pred = np.random.randn(8, 3)
    y_true = np.random.randn(8, 3)
    mse(y_pred, y_true)
    grad = mse.backward()
    assert grad.shape == y_pred.shape


def test_crossentropy_with_integer_labels():
    ce = CrossEntropy()
    y_pred = np.array([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
    y_true = np.array([1, 2])
    loss = ce(y_pred, y_true)
    assert loss > 0


def test_crossentropy_perfect_predictions():
    ce = CrossEntropy()
    y_pred = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = np.array([0, 1])
    loss = ce(y_pred, y_true)
    assert loss == pytest.approx(0.0, abs=1e-3)


def test_crossentropy_backward_shape():
    ce = CrossEntropy()
    y_pred = np.array([[0.2, 0.5, 0.3]])
    y_true = np.array([1])
    ce(y_pred, y_true)
    grad = ce.backward()
    assert grad.shape == y_pred.shape


def test_binary_crossentropy_known():
    bce = BinaryCrossEntropy()
    y_pred = np.array([[1.0]])
    y_true = np.array([[1.0]])
    loss = bce(y_pred, y_true)
    assert loss == pytest.approx(0.0, abs=1e-3)


def test_binary_crossentropy_backward_shape():
    bce = BinaryCrossEntropy()
    y_pred = np.array([[0.7], [0.3]])
    y_true = np.array([[1.0], [0.0]])
    bce(y_pred, y_true)
    grad = bce.backward()
    assert grad.shape == y_pred.shape
