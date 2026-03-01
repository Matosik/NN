"""Optimizers for training neural networks."""

import numpy as np
from typing import Dict


class Optimizer:
    """Base class for optimizers."""

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum.

    Parameters
    ----------
    lr:
        Learning rate.
    momentum:
        Momentum factor (0 disables momentum).
    weight_decay:
        L2 regularization coefficient.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: Dict[int, Dict[str, np.ndarray]] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        pid = id(params)
        if pid not in self._velocity:
            self._velocity[pid] = {k: np.zeros_like(v) for k, v in params.items()}

        for k in params:
            g = grads[k] + self.weight_decay * params[k]
            if self.momentum > 0:
                self._velocity[pid][k] = self.momentum * self._velocity[pid][k] + g
                params[k] -= self.lr * self._velocity[pid][k]
            else:
                params[k] -= self.lr * g


class Adam(Optimizer):
    """Adam optimizer.

    Parameters
    ----------
    lr:
        Learning rate (step size).
    beta1:
        Exponential decay rate for the first moment estimates.
    beta2:
        Exponential decay rate for the second moment estimates.
    eps:
        Small constant for numerical stability.
    weight_decay:
        L2 regularization coefficient.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m: Dict[int, Dict[str, np.ndarray]] = {}
        self._v: Dict[int, Dict[str, np.ndarray]] = {}
        self._t: Dict[int, int] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        pid = id(params)
        if pid not in self._m:
            self._m[pid] = {k: np.zeros_like(v) for k, v in params.items()}
            self._v[pid] = {k: np.zeros_like(v) for k, v in params.items()}
            self._t[pid] = 0

        self._t[pid] += 1
        t = self._t[pid]
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** t) / (1.0 - self.beta1 ** t)

        for k in params:
            g = grads[k] + self.weight_decay * params[k]
            self._m[pid][k] = self.beta1 * self._m[pid][k] + (1.0 - self.beta1) * g
            self._v[pid][k] = self.beta2 * self._v[pid][k] + (1.0 - self.beta2) * g ** 2
            params[k] -= lr_t * self._m[pid][k] / (np.sqrt(self._v[pid][k]) + self.eps)
