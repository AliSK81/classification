import numpy as np


class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, b_input):
        self.d_weights = np.dot(self.inputs.T, b_input)
        self.d_biases = np.sum(b_input, axis=0, keepdims=True)
        self.b_output = np.dot(b_input, self.weights.T)
        return self.b_output
