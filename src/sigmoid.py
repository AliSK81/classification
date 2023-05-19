import numpy as np


class Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, b_input):
        self.b_output = b_input * self.output * (1 - self.output)
