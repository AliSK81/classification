from src.data_loader import load_data
from src.som import SOM
from src.util import extract_features, show_confusion_matrix


def main():
    train_loader, test_loader = load_data(batch_size=1)

    x_train, y_train = extract_features(train_loader)
    x_test, y_test = extract_features(test_loader)

    scenarios = [
        {"name": "Scenario a", "num_neurons": (10, 1), "neighborhood_radius": 1},
        {"name": "Scenario b", "num_neurons": (10, 1), "neighborhood_radius": 3},
        {"name": "Scenario c", "num_neurons": (5, 2), "neighborhood_radius": 1}
    ]

    for scenario in scenarios:
        print(f"Running {scenario['name']}")
        som = SOM(
            input_dim=x_train.shape[1],
            grid_dimensions=scenario['num_neurons'],
            neighborhood_radius=scenario['neighborhood_radius']
        )
        som.train(x_train, y_train, num_epochs=20, learning_rate=0.1)
        y_predict = som.predict(x_test)

        show_confusion_matrix(y_test, y_predict, scenario['name'])


if __name__ == '__main__':
    main()


