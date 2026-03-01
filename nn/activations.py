"""Activation functions for neural networks."""

import numpy as np


class Activation:
    """Base class for activation functions."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class ReLU(Activation):
    """Rectified Linear Unit activation: f(x) = max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return np.maximum(0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (self._input > 0).astype(float)


class Sigmoid(Activation):
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x))."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._output = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self._output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._output * (1.0 - self._output)


class Tanh(Activation):
    """Hyperbolic tangent activation: f(x) = tanh(x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._output = np.tanh(x)
        return self._output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._output ** 2)


class Softmax(Activation):
    """Softmax activation: f(x)_i = exp(x_i) / sum(exp(x_j))."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        self._output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self._output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Jacobian-vector product for softmax
        s = self._output
        return s * (grad - np.sum(grad * s, axis=-1, keepdims=True))


# Convenience instances
relu = ReLU()
sigmoid = Sigmoid()
tanh = Tanh()
softmax = Softmax()
