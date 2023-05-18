# torch is just for the feature extractor and the dataset (NOT FOR IMPLEMENTING NEURAL NETWORKS!)
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

# sklearn is just for evaluation (NOT FOR IMPLEMENTING NEURAL NETWORKS!)

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# You should define x_train and y_train


# Define the feature extractor
feature_extractor = resnet34(pretrained=True)
num_features = feature_extractor.fc.in_features
feature_extractor.fc = nn.Identity()

feature_extractor.eval()
with torch.no_grad():
    features_list = []
    y_train = []
    i = 0
    for images, labels in train_loader:
        i += 1
        if i == 2:
            break
        features = feature_extractor(images)
        features_list.append(features)
        y_train.append(labels)

    features_train = torch.cat(features_list, dim=0)
    x_train = torch.stack(features_list).numpy()

y_train = torch.cat(y_train, dim=0).numpy()


class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, b_input):
        self.b_weights = np.dot(self.inputs.T, b_input)
        self.b_biases = np.sum(b_input, axis=0, keepdims=True)
        self.b_output = np.dot(b_input, self.weights.T)


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, b_input):
        self.b_output = b_input * (self.inputs > 0)


class Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, b_input):
        self.b_output = b_input * self.output * (1 - self.output)


class Softmax:
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def backward(self, b_input):
        self.b_output = b_input


class Categorical_Cross_Entropy_loss:
    def forward(self, softmax_output, class_label):
        self.softmax_output = softmax_output
        self.class_label = class_label
        self.loss = -np.mean(np.log(self.softmax_output[np.arange(len(class_label)), class_label]))

    def backward(self, softmax_output, class_label):
        self.b_output = softmax_output
        self.b_output[np.arange(len(class_label)), class_label] -= 1
        self.b_output /= len(class_label)


class SGD:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update(self, layer):
        layer.weights -= self.learning_rate * layer.b_weights
        layer.biases -= self.learning_rate * layer.b_biases


feature_extractor = resnet34(pretrained=True)
d = feature_extractor.fc.in_features

Layer1 = Dense(d, 20)  # d is the output dimension of feature extractor
Act1 = ReLU()
Layer2 = Dense(20, 10)
Act2 = Softmax()
Loss = Categorical_Cross_Entropy_loss()
Optimizer = SGD(learning_rate=0.001)

# Main Loop of Training
for epoch in range(20):
    # forward
    Layer1.forward(x_train)
    Act1.forward(Layer1.output)
    Layer2.forward(Act1.output)
    Act2.forward(Layer2.output)

    y_1hot = np.zeros((y_train.shape[0], 10))
    y_1hot[np.arange(y_train.shape[0]), y_train] = 1

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

# Confusion Matrix for the training set
cm_train = confusion_matrix(y_train, y_predict)
plt.subplots(figsize=(10, 6))
sb.heatmap(cm_train, annot=True, fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for the training set")
plt.show()

# Confusion Matrix for the test set
# TODO
