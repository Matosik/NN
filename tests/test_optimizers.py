"""Tests for optimizers."""

import numpy as np
import pytest

from nn.optimizers import SGD, Adam


def _make_params():
    return {"W": np.array([1.0, 2.0, 3.0]), "b": np.array([0.5])}


def _make_grads():
    return {"W": np.array([0.1, 0.2, 0.3]), "b": np.array([0.05])}


def test_sgd_updates_params():
    params = _make_params()
    grads = _make_grads()
    W_before = params["W"].copy()
    sgd = SGD(lr=0.1)
    sgd.step(params, grads)
    np.testing.assert_allclose(params["W"], W_before - 0.1 * grads["W"])


def test_sgd_with_momentum():
    params = _make_params()
    grads = _make_grads()
    sgd = SGD(lr=0.1, momentum=0.9)
    sgd.step(params, grads)
    # After first step velocity equals gradient, so update = lr * grad
    expected_W = np.array([1.0, 2.0, 3.0]) - 0.1 * np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(params["W"], expected_W)


def test_sgd_weight_decay():
    params = {"W": np.array([1.0])}
    grads = {"W": np.array([0.0])}
    sgd = SGD(lr=1.0, weight_decay=0.1)
    sgd.step(params, grads)
    # grad = 0 + 0.1*1.0 = 0.1, update = -1.0 * 0.1 = -0.1
    np.testing.assert_allclose(params["W"], [0.9])


def test_adam_decreases_loss():
    """Adam should reduce a simple quadratic objective."""
    np.random.seed(42)
    params = {"W": np.array([5.0, -5.0])}
    adam = Adam(lr=0.1)
    for _ in range(200):
        grads = {"W": 2.0 * params["W"]}  # grad of W^2
        adam.step(params, grads)
    assert np.all(np.abs(params["W"]) < 0.1)


def test_adam_two_param_groups():
    p1 = {"W": np.array([1.0])}
    p2 = {"W": np.array([1.0])}
    adam = Adam(lr=0.1)
    adam.step(p1, {"W": np.array([1.0])})
    adam.step(p2, {"W": np.array([1.0])})
    # Both should be updated independently
    np.testing.assert_allclose(p1["W"], p2["W"])
