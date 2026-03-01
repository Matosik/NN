"""Example: regression on a sine wave."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn import Sequential, Dense, MSE, Adam
from nn.layers import ActivationLayer
from nn.activations import Tanh

np.random.seed(0)
X = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)
y = np.sin(X)

model = Sequential([
    Dense(1, 32),
    ActivationLayer(Tanh()),
    Dense(32, 32),
    ActivationLayer(Tanh()),
    Dense(32, 1),
])
model.compile(loss=MSE(), optimizer=Adam(lr=0.005))
history = model.fit(X, y, epochs=500, verbose=True, batch_size=64)

preds = model.predict(X)
mse = np.mean((preds - y) ** 2)
print(f"\nFinal MSE on sine wave: {mse:.6f}")
