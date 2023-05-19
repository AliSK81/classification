import numpy as np


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, b_input):
        self.b_output = b_input * (self.inputs > 0).astype(int)
        return self.b_output