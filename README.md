# NN — Neural Networks and Deep Learning

A lightweight neural network library built on top of NumPy.  
It provides all the building blocks needed to construct, train, and evaluate
feed-forward deep learning models from scratch.

---

## Features

| Component | Options |
|-----------|---------|
| **Activations** | `ReLU`, `Sigmoid`, `Tanh`, `Softmax` |
| **Layers** | `Dense` (fully-connected), `Dropout`, `ActivationLayer` |
| **Losses** | `MSE`, `CrossEntropy`, `BinaryCrossEntropy` |
| **Optimizers** | `SGD` (with momentum & weight-decay), `Adam` |
| **Model** | `Sequential` |

---

## Installation

```bash
pip install numpy   # only dependency
```

---

## Quick start

```python
import numpy as np
from nn import Sequential, Dense, ReLU, Sigmoid, BinaryCrossEntropy, Adam
from nn.layers import ActivationLayer

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]],         dtype=float)

model = Sequential([
    Dense(2, 8),
    ActivationLayer(ReLU()),
    Dense(8, 1),
    ActivationLayer(Sigmoid()),
])
model.compile(loss=BinaryCrossEntropy(), optimizer=Adam(lr=0.05))
history = model.fit(X, y, epochs=1000, batch_size=4)

preds = model.predict(X)
print((preds > 0.5).astype(int).T)
# [[0 1 1 0]]
```

---

## API

### `Sequential`

```python
model = Sequential(layers=[...])   # or Sequential() then .add()
model.compile(loss, optimizer)
history = model.fit(X, y, epochs=100, batch_size=32, validation_data=(X_val, y_val))
preds   = model.predict(X)
loss    = model.evaluate(X, y)
```

### Layers

```python
Dense(in_features, out_features)   # learnable W, b
Dropout(p=0.5)                     # inverted dropout; set .training = False for inference
ActivationLayer(activation)        # wraps any Activation as a Layer
```

### Activations

```python
ReLU()     # max(0, x)
Sigmoid()  # 1 / (1 + exp(-x))
Tanh()     # tanh(x)
Softmax()  # exp(x_i) / Σ exp(x_j)
```

### Losses

```python
MSE()                   # mean squared error
CrossEntropy()          # multi-class; expects probabilities + integer / one-hot labels
BinaryCrossEntropy()    # binary; expects probabilities + {0,1} labels
```

### Optimizers

```python
SGD(lr=0.01, momentum=0.0, weight_decay=0.0)
Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
```

---

## Examples

```bash
python examples/xor.py            # XOR classification
python examples/sine_regression.py  # sine-wave regression
```

---

## Tests

```bash
python -m pytest tests/ -v
```
