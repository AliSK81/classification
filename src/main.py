import numpy as np
import seaborn as sb
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import resnet34

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

# Define the feature extractor
feature_extractor = resnet34(pretrained=True)
num_features = feature_extractor.fc.in_features
feature_extractor.fc = nn.Identity()


class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, d_outputs):
        d_weights = np.dot(self.inputs.T, d_outputs)
        d_biases = np.sum(d_outputs, axis=0, keepdims=True)
        d_inputs = np.dot(d_outputs, self.weights.T)

        return d_weights, d_biases, d_inputs


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, d_outputs):
        return d_outputs * (self.inputs > 0).astype(int)


class Softmax:
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def backward(self, d_outputs, y_true):
        batch_size = d_outputs.shape[0]
        return (d_outputs - y_true) / batch_size


class CategoricalCrossEntropyLoss:
    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / batch_size

    def backward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        return (y_pred - y_true) / batch_size


class SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update(self, layer, d_weights, d_biases):
        layer.weights -= self.learning_rate * d_weights
        layer.biases -= self.learning_rate * d_biases


d = num_features
Layer1 = Dense(d, 20)
Act1 = ReLU()
Layer2 = Dense(20, 10)
Act2 = Softmax()
Loss = CategoricalCrossEntropyLoss()
Optimizer = SGD(learning_rate=0.001)

num_classes = 10

# Main Loop of Training
for epoch in range(20):

    feature_extractor.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            features = feature_extractor(images)

            x_train = features.numpy()
            y_train = labels.numpy()

            y_1hot = np.eye(num_classes)[y_train]

            # forward
            output1 = Layer1.forward(x_train)
            output2 = Act1.forward(output1)
            output3 = Layer2.forward(output2)
            y_pred = Act2.forward(output3)
            loss = Loss.forward(y_pred, y_1hot)

            # Report
            y_predict = np.argmax(y_pred, axis=1)
            accuracy = np.mean(y_train == y_predict)
            print(f'Epoch:{epoch}')
            print(f'Loss: {loss}')
            print(f'Accuracy: {accuracy}')
            print('--------------------------')

            # backward
            d_loss = Loss.backward(y_pred, y_1hot)
            d_output3 = Act2.backward(d_loss, y_1hot)
            d_output2, d_biases2, d_output1 = Layer2.backward(d_output3)
            d_output1 = Act1.backward(d_output1)
            d_output0, d_biases1, _ = Layer1.backward(d_output1)

            # update params
            Optimizer.update(Layer1, d_output0, d_biases1)
            Optimizer.update(Layer2, d_output2, d_biases2)

# Confusion Matrix for the training set
cm_train = confusion_matrix(y_train, y_predict)
plt.subplots(figsize=(10, 6))
sb.heatmap(cm_train, annot=True, fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for the training set")
plt.show()
