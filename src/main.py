from torch import nn
from torchvision.models import resnet34

from evaluator import evaluate_model
from sgd import SGD
from src.custom_model import CustomModel
from src.data_loader import load_data
from src.dense import Dense
from src.relu import ReLU
from src.softmax import Softmax
from src.trainer import train_model


def main():
    train_loader, test_loader = load_data(batch_size=100)
    n_classes = len(train_loader.dataset.classes)

    feature_extractor = resnet34(pretrained=True)
    n_features = feature_extractor.fc.in_features
    feature_extractor.fc = nn.Identity()

    model = CustomModel(layers=[
        Dense(n_features, 20),
        ReLU(),
        Dense(20, n_classes),
        Softmax()
    ])
    optimizer = SGD(learning_rate=0.001)

    epochs = 10
    train_model(model, optimizer, feature_extractor, train_loader, epochs, n_classes)
    evaluate_model(model, feature_extractor, train_loader, "training")
    evaluate_model(model, feature_extractor, test_loader, "testing")


if __name__ == "__main__":
    main()
