import seaborn as sb
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn
from torchvision.models import resnet34


def show_confusion_matrix(y_true, y_pred, scenario_name):
    cm_train = confusion_matrix(y_true, y_pred)
    plt.subplots(figsize=(10, 10))
    sb.heatmap(cm_train, annot=True, fmt="g")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for the training set ({scenario_name})")
    plt.show()


def extract_features(loader):
    feature_extractor = resnet34(pretrained=True)
    feature_extractor.fc = nn.Identity()
    feature_extractor.eval()

    with torch.no_grad():
        features_list = []
        labels_list = []
        for images, labels in loader:
            features = feature_extractor(images)
            features_list.append(features)
            labels_list.append(labels)
    return _as_numpy(features_list), _as_numpy(labels_list)


def _as_numpy(list):
    return torch.cat(list, dim=0).numpy()