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
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, b_input):
        self.d_weights = np.dot(self.inputs.T, b_input)
        self.d_biases = np.sum(b_input, axis=0, keepdims=True)
        self.b_output = np.dot(b_input, self.weights.T)


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, b_input):
        self.b_output = b_input * (self.inputs > 0).astype(int)


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


class CategoricalCrossEntropyLoss:
    def forward(self, softmax_output, class_label):
        batch_size = softmax_output.shape[0]
        self.output = -np.sum(class_label * np.log(softmax_output + 1e-9)) / batch_size
        return self.output

    def backward(self, softmax_output, class_label):
        self.b_output = - class_label / softmax_output

        # batch_size = softmax_output.shape[0]
        # self.b_output = (softmax_output - class_label) / batch_size


class SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update(self, layer):
        layer.weights -= self.learning_rate * layer.d_weights
        layer.biases -= self.learning_rate * layer.d_biases


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
            Layer1.forward(x_train)
            Act1.forward(Layer1.output)
            Layer2.forward(Act1.output)
            Act2.forward(Layer2.output)
            loss = Loss.forward(Act2.output, y_1hot)

            # Report
            y_predict = np.argmax(Act2.output, axis=1)
            accuracy = np.mean(y_train == y_predict)
            print(f'Epoch:{epoch}')
            print(f'Loss: {loss}')
            print(f'Accuracy: {accuracy}')
            print('--------------------------')

            # backward
            Loss.backward(Act2.output, y_1hot)
            Act2.backward(Loss.b_output)
            Layer2.backward(Act2.b_output)
            Act1.backward(Layer2.b_output)
            Layer1.backward(Act1.b_output)

            # update params
            Optimizer.update(Layer1)
            Optimizer.update(Layer2)

feature_extractor.eval()
with torch.no_grad():
    y_test_all = []
    y_predict_all = []
    for images, labels in test_loader:
        features = feature_extractor(images)

        x_test = features.numpy()
        y_test = labels.numpy()

        # forward
        Layer1.forward(x_test)
        Act1.forward(Layer1.output)
        Layer2.forward(Act1.output)
        Act2.forward(Layer2.output)

        # Report
        y_predict = np.argmax(Act2.output, axis=1)

        y_test_all.extend(y_test)
        y_predict_all.extend(y_predict)

# Confusion Matrix for the testing set
cm_test = confusion_matrix(y_test_all, y_predict_all)
plt.subplots(figsize=(10, 6))
sb.heatmap(cm_test, annot=True, fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for the testing set")
plt.show()
