import numpy as np
import torch
from tqdm import tqdm

from src.dense import Dense


def train_model(model, optimizer, feature_extractor, train_loader, epochs, n_classes):
    for epoch in range(epochs):
        feature_extractor.eval()

        with torch.no_grad(), tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='batch') as pbar:
            for images, labels in train_loader:
                features = feature_extractor(images)

                x_train = features.numpy()
                y_train = labels.numpy()

                y_1hot = np.eye(n_classes)[y_train]

                # forward
                outputs = model.forward(x_train)
                loss = model.loss.forward(outputs, y_1hot)

                # backward
                model.loss.backward(outputs, y_1hot)
                b_output = model.loss.b_output
                model.backward(b_output)

                # update params
                for layer in model.layers:
                    if isinstance(layer, Dense):
                        optimizer.update(layer)

                # Update the progress bar
                pbar.update(1)

            # Report
            y_predict = model.predict(x_train)
            accuracy = np.mean(y_train == y_predict)
            print(f'\nEpoch:{epoch}')
            print(f'Loss: {loss}')
            print(f'Accuracy: {accuracy}')
            print('--------------------------')
