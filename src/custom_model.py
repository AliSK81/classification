import numpy as np

from src.cce import CategoricalCrossEntropyLoss


class CustomModel:
    def __init__(self, layers):
        self.layers = layers
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
