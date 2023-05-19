from torch import nn
from torchvision.models import resnet34

from evaluator import evaluate_model
from sgd import SGD
from src.custom_model import CustomModel
from src.data_loader import load_data
from src.trainer import train_model


def main():
    train_loader, test_loader = load_data(batch_size=100)

    feature_extractor = resnet34(pretrained=True)
    n_features = feature_extractor.fc.in_features
    feature_extractor.fc = nn.Identity()

    model = CustomModel(n_features=n_features, n_classes=len(train_loader.dataset.classes))
    optimizer = SGD(learning_rate=0.001)

    epochs = 5
    train_model(model, optimizer, feature_extractor, train_loader, epochs)
    evaluate_model(model, feature_extractor, train_loader, "training")
    evaluate_model(model, feature_extractor, test_loader, "testing")


if __name__ == "__main__":
    main()
