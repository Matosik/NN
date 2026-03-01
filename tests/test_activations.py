"""Tests for activation functions."""

import numpy as np
import pytest

from nn.activations import ReLU, Sigmoid, Tanh, Softmax


def test_relu_forward():
    relu = ReLU()
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = relu.forward(x)
    np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 1.0, 2.0])


def test_relu_backward():
    relu = ReLU()
    x = np.array([-1.0, 0.0, 1.0])
    relu.forward(x)
    grad = relu.backward(np.ones(3))
    np.testing.assert_array_equal(grad, [0.0, 0.0, 1.0])


def test_sigmoid_range():
    sigmoid = Sigmoid()
    x = np.linspace(-10, 10, 100)
    out = sigmoid.forward(x)
    assert np.all(out > 0) and np.all(out < 1)


def test_sigmoid_backward():
    sigmoid = Sigmoid()
    x = np.array([0.0])
    out = sigmoid.forward(x)
    grad = sigmoid.backward(np.ones(1))
    expected = out * (1 - out)
    np.testing.assert_allclose(grad, expected)


def test_tanh_range():
    tanh = Tanh()
    x = np.linspace(-10, 10, 100)
    out = tanh.forward(x)
    assert np.all(out >= -1) and np.all(out <= 1)


def test_tanh_backward():
    tanh = Tanh()
    x = np.array([0.0])
    out = tanh.forward(x)
    grad = tanh.backward(np.ones(1))
    expected = 1 - out ** 2
    np.testing.assert_allclose(grad, expected)


def test_softmax_sums_to_one():
    softmax = Softmax()
    x = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    out = softmax.forward(x)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(2), atol=1e-7)


def test_softmax_backward_shape():
    softmax = Softmax()
    x = np.random.randn(4, 5)
    softmax.forward(x)
    grad = softmax.backward(np.ones_like(x))
    assert grad.shape == x.shape
