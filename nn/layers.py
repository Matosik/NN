"""Neural network layers."""

import numpy as np
from .activations import Activation


class Layer:
    """Base class for all layers."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    @property
    def parameters(self):
        return {}

    @property
    def gradients(self):
        return {}


class Dense(Layer):
    """Fully-connected (dense) layer.

    Computes output = x @ W + b, where W has shape (in_features, out_features)
    and b has shape (out_features,).
    """

    def __init__(self, in_features: int, out_features: int):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self._dW = np.zeros_like(self.W)
        self._db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self._dW = self._input.T @ grad
        self._db = grad.sum(axis=0)
        return grad @ self.W.T

    @property
    def parameters(self):
        return {"W": self.W, "b": self.b}

    @property
    def gradients(self):
        return {"W": self._dW, "b": self._db}


class ActivationLayer(Layer):
    """Wraps an Activation function as a Layer."""

    def __init__(self, activation: Activation):
        self._activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._activation.forward(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self._activation.backward(grad)


class Dropout(Layer):
    """Dropout layer for regularization.

    During training, randomly zeros elements with probability ``p``.
    During inference the layer is a no-op.
    """

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.training = True
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training and self.p > 0:
            self._mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            return x * self._mask
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.training and self._mask is not None:
            return grad * self._mask
        return grad
