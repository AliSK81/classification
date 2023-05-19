import numpy as np


class CategoricalCrossEntropyLoss:
    def forward(self, softmax_output, class_label):
        batch_size = softmax_output.shape[0]
        self.output = -np.sum(class_label * np.log(softmax_output + 1e-9)) / batch_size
        return self.output

    def backward(self, softmax_output, class_label):
        self.b_output = - class_label / softmax_output
        return self.b_output
