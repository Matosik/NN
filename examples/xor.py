"""Example: XOR problem solved with a small neural network."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn import Sequential, Dense, Sigmoid, BinaryCrossEntropy, Adam
from nn.layers import ActivationLayer

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

np.random.seed(42)
model = Sequential([
    Dense(2, 8),
    ActivationLayer(Sigmoid()),
    Dense(8, 1),
    ActivationLayer(Sigmoid()),
])
model.compile(loss=BinaryCrossEntropy(), optimizer=Adam(lr=0.05))
history = model.fit(X, y, epochs=1000, verbose=True, batch_size=4)

preds = model.predict(X)
print("\nXOR predictions (threshold 0.5):")
for xi, pi in zip(X, preds):
    print(f"  {xi.astype(int)} -> {int(pi[0] > 0.5)}  (raw: {pi[0]:.4f})")
