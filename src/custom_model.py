import numpy as np

from src.cce import CategoricalCrossEntropyLoss
from src.dense import Dense
from src.relu import ReLU
from src.softmax import Softmax


class CustomModel:
    def __init__(self, n_features, n_classes):
        self.n_classes = n_classes
        self.layers = [
            Dense(n_features, 20),
            ReLU(),
            Dense(20, n_classes),
            Softmax()
        ]
        self.loss = CategoricalCrossEntropyLoss()

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, b_input):
        for layer in reversed(self.layers):
            b_input = layer.backward(b_input)
        return b_input

    def predict(self, inputs):
        output = self.forward(inputs)
        return np.argmax(output, axis=1)
