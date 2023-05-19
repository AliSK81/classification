import seaborn as sb
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm


def evaluate_model(model, feature_extractor, data_loader, evaluation_name):
    feature_extractor.eval()
    with torch.no_grad():
        y_true_all = []
        y_predict_all = []
        for images, labels in tqdm(data_loader,
                                   total=len(data_loader), desc=f"Evaluating {evaluation_name} set", unit='batch'):
            features = feature_extractor(images)

            x = features.numpy()
            y_true = labels.numpy()

            y_predict = model.predict(x)

            y_true_all.extend(y_true)
            y_predict_all.extend(y_predict)

    cm = confusion_matrix(y_true_all, y_predict_all)
    plt.subplots(figsize=(10, 6))
    sb.heatmap(cm, annot=True, fmt='g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for the {evaluation_name} set")
    plt.show()

    accuracy = accuracy_score(y_true_all, y_predict_all)
    f1 = f1_score(y_true_all, y_predict_all, average='macro')
    print(f'\n{evaluation_name} Accuracy: {accuracy}')
    print(f'{evaluation_name} F1 Score: {f1}')
    return accuracy
