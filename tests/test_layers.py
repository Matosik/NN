"""Tests for neural network layers."""

import numpy as np
import pytest

from nn.layers import Dense, Dropout, ActivationLayer
from nn.activations import ReLU


def test_dense_forward_shape():
    layer = Dense(4, 3)
    x = np.random.randn(10, 4)
    out = layer.forward(x)
    assert out.shape == (10, 3)


def test_dense_backward_shape():
    layer = Dense(4, 3)
    x = np.random.randn(10, 4)
    layer.forward(x)
    grad_in = np.random.randn(10, 3)
    grad_out = layer.backward(grad_in)
    assert grad_out.shape == (10, 4)
    assert layer.gradients["W"].shape == (4, 3)
    assert layer.gradients["b"].shape == (3,)


def test_dense_gradient_numerical():
    """Finite-difference gradient check for Dense layer."""
    np.random.seed(0)
    layer = Dense(3, 2)
    x = np.random.randn(5, 3)
    eps = 1e-5

    out = layer.forward(x)
    grad_output = np.ones_like(out)
    grad_x = layer.backward(grad_output)

    # Numerical gradient w.r.t. input
    num_grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xp = x.copy(); xp[i, j] += eps
            xm = x.copy(); xm[i, j] -= eps
            layer._input = xp
            fp = (xp @ layer.W + layer.b).sum()
            layer._input = xm
            fm = (xm @ layer.W + layer.b).sum()
            num_grad[i, j] = (fp - fm) / (2 * eps)

    np.testing.assert_allclose(grad_x, num_grad, rtol=1e-4)


def test_dropout_training_zeros():
    layer = Dropout(p=0.99)
    layer.training = True
    x = np.ones((1000, 10))
    out = layer.forward(x)
    # With p=0.99 almost all outputs should be zero
    zero_fraction = np.mean(out == 0)
    assert zero_fraction > 0.9


def test_dropout_inference_passthrough():
    layer = Dropout(p=0.5)
    layer.training = False
    x = np.ones((10, 10))
    out = layer.forward(x)
    np.testing.assert_array_equal(out, x)


def test_dropout_invalid_p():
    with pytest.raises(ValueError):
        Dropout(p=1.5)


def test_activation_layer():
    relu = ReLU()
    layer = ActivationLayer(relu)
    x = np.array([[-1.0, 0.0, 1.0]])
    out = layer.forward(x)
    np.testing.assert_array_equal(out, [[0.0, 0.0, 1.0]])
    grad = layer.backward(np.ones_like(x))
    np.testing.assert_array_equal(grad, [[0.0, 0.0, 1.0]])
