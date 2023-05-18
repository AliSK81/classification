# torch is just for the feature extractor and the dataset (NOT FOR IMPLEMENTING NEURAL NETWORKS!)
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
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


# You should define x_train and y_train

class Dense:
    def __init__(self, n_inputs, n_neurons):
        pass

    #  TODO: Define initial weight and bias

    def forward(self, inputs):
        pass

    #  TODO: Define input and output

    def backward(self, b_input):
        pass


#  TODO: Weight and bias gradients

class ReLU:
    def forward(self, inputs):
        pass

    #  TODO: Implement the ReLU formula

    def backward(self, b_input):
        pass

    #  TODO: Implement the ReLU derivative with respect to the input


class Sigmoid:
    def forward(self, inputs):
        pass

    #  TODO: Implement the sigmoid formula

    def backward(self, b_input):
        pass


#  TODO: Implement the sigmoid derivative with respect to the input
class Softmax:
    def forward(self, inputs):
        pass

    #  TODO: Implement the softmax formula

    def backward(self, b_input):
        #  TODO: Implement the softmax derivative with respect to the input

        pass


class Categorical_Cross_Entropy_loss:
    def forward(self, softmax_output, class_label):
        pass
        # TODO: Implement the CCE loss formula

    def backward(self, softmax_output, class_label):
        pass
        # TODO: Implement the CCE loss derivative with respect to predicted label

    class SGD:
        def __init__(self, learning_rate=0.001):
            self.learning_rate = learning_rate

        def update(self, layer):
            pass
    # TODO: Update layer params based on gradient descent rule


feature_extractor = resnet34(pretrained=True)
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
