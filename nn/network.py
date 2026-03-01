"""Sequential neural network model."""

import numpy as np
from typing import List, Optional
from .layers import Layer, Dropout
from .losses import Loss
from .optimizers import Optimizer


class Sequential:
    """A sequential container of layers.

    Layers are stacked in the order they are provided and executed from
    first to last during the forward pass.

    Example
    -------
    >>> from nn import Sequential, Dense, ReLU, Sigmoid, MSE, Adam
    >>> model = Sequential([Dense(2, 4), ReLU(), Dense(4, 1), Sigmoid()])
    >>> model.compile(loss=MSE(), optimizer=Adam(lr=0.01))
    >>> history = model.fit(X_train, y_train, epochs=100)
    """

    def __init__(self, layers: Optional[List[Layer]] = None):
        self.layers: List[Layer] = layers or []
        self._loss: Optional[Loss] = None
        self._optimizer: Optional[Optimizer] = None

    def add(self, layer: Layer) -> "Sequential":
        """Append a layer to the model."""
        self.layers.append(layer)
        return self

    def compile(self, loss: Loss, optimizer: Optimizer) -> None:
        """Set the loss function and optimizer."""
        self._loss = loss
        self._optimizer = optimizer

    # ------------------------------------------------------------------
    # Training / inference helpers
    # ------------------------------------------------------------------

    def _set_training(self, training: bool) -> None:
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run a forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def _backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update(self) -> None:
        for layer in self.layers:
            if layer.parameters:
                self._optimizer.step(layer.parameters, layer.gradients)

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
        validation_data=None,
    ) -> dict:
        """Train the model.

        Parameters
        ----------
        X:
            Input data, shape ``(n_samples, n_features)``.
        y:
            Target values, shape ``(n_samples, ...)`` or ``(n_samples,)``.
        epochs:
            Number of training epochs.
        batch_size:
            Mini-batch size.
        verbose:
            Whether to print loss after each epoch.
        validation_data:
            Optional ``(X_val, y_val)`` tuple for validation.

        Returns
        -------
        dict
            Training history with keys ``"loss"`` (and ``"val_loss"`` when
            validation data is provided).
        """
        if self._loss is None or self._optimizer is None:
            raise RuntimeError("Call compile() before fit().")

        n = X.shape[0]
        history = {"loss": []}
        if validation_data is not None:
            history["val_loss"] = []

        for epoch in range(1, epochs + 1):
            self._set_training(True)
            indices = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                y_pred = self.forward(X_batch)
                loss = self._loss(y_pred, y_batch)
                epoch_loss += loss
                n_batches += 1

                grad = self._loss.backward()
                self._backward(grad)
                self._update()

            avg_loss = epoch_loss / n_batches
            history["loss"].append(avg_loss)

            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                msg = f"Epoch {epoch:4d}/{epochs} — loss: {avg_loss:.6f}"
                if "val_loss" in history:
                    msg += f"  val_loss: {history['val_loss'][-1]:.6f}"
                print(msg)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on ``X`` (training mode is disabled)."""
        self._set_training(False)
        return self.forward(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the loss on ``(X, y)`` without updating weights."""
        self._set_training(False)
        y_pred = self.forward(X)
        return self._loss(y_pred, y)
