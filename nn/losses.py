"""Loss functions for neural networks."""

import numpy as np


class Loss:
    """Base class for loss functions."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.forward(y_pred, y_true)


class MSE(Loss):
    """Mean Squared Error loss: L = mean((y_pred - y_true)^2)."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._diff = y_pred - y_true
        self._n = y_pred.shape[0]
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        return 2.0 * self._diff / self._diff.size


class CrossEntropy(Loss):
    """Cross-Entropy loss for multi-class classification.

    Expects ``y_pred`` to be class probabilities (e.g. after Softmax) and
    ``y_true`` to be integer class indices or one-hot encoded vectors.
    """

    def __init__(self, eps: float = 1e-12):
        self._eps = eps

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._n = y_pred.shape[0]
        probs = np.clip(y_pred, self._eps, 1.0 - self._eps)
        self._probs = probs

        if y_true.ndim == 1:
            # Integer class labels
            self._y_true_onehot = np.zeros_like(probs)
            self._y_true_onehot[np.arange(self._n), y_true.astype(int)] = 1.0
        else:
            self._y_true_onehot = y_true.astype(float)

        return float(-np.sum(self._y_true_onehot * np.log(probs)) / self._n)

    def backward(self) -> np.ndarray:
        return -(self._y_true_onehot / self._probs) / self._n


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy loss for binary classification.

    Expects ``y_pred`` to be probabilities in (0, 1) (e.g. after Sigmoid) and
    ``y_true`` to be binary labels (0 or 1).
    """

    def __init__(self, eps: float = 1e-12):
        self._eps = eps

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._n = y_pred.shape[0]
        probs = np.clip(y_pred, self._eps, 1.0 - self._eps)
        self._probs = probs
        self._y_true = y_true.astype(float)
        return float(
            -np.mean(
                self._y_true * np.log(probs) + (1.0 - self._y_true) * np.log(1.0 - probs)
            )
        )

    def backward(self) -> np.ndarray:
        return (
            -(self._y_true / self._probs - (1.0 - self._y_true) / (1.0 - self._probs))
            / self._n
        )
