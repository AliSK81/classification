import numpy as np


class Softmax:
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self.output

    def backward(self, b_input):
        num_classes = b_input.shape[1]

        jacobian_matrix = -self.output[:, :, np.newaxis] * self.output[:, np.newaxis, :]
        identity = np.eye(num_classes)
        jacobian_matrix += identity[np.newaxis, :, :] * self.output[:, np.newaxis, :]

        self.b_output = np.einsum('bij,bj->bi', jacobian_matrix, b_input)
        return self.b_output
