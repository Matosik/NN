"""Neural Networks and Deep Learning library.

Provides building blocks for constructing and training feed-forward neural
networks using NumPy.

Modules
-------
activations : ReLU, Sigmoid, Tanh, Softmax
layers      : Dense, Dropout, ActivationLayer
losses      : MSE, CrossEntropy, BinaryCrossEntropy
optimizers  : SGD, Adam
network     : Sequential
"""

from .activations import ReLU, Sigmoid, Tanh, Softmax
from .layers import Dense, Dropout, ActivationLayer
from .losses import MSE, CrossEntropy, BinaryCrossEntropy
from .optimizers import SGD, Adam
from .network import Sequential

__all__ = [
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Dense",
    "Dropout",
    "ActivationLayer",
    "MSE",
    "CrossEntropy",
    "BinaryCrossEntropy",
    "SGD",
    "Adam",
    "Sequential",
]
