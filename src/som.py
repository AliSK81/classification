import numpy as np


class SOM:
    def __init__(self, input_dim, grid_dimensions, neighborhood_radius):
        self.input_dim = input_dim
        self.grid_dimensions = grid_dimensions
        self.neighborhood_radius = neighborhood_radius
        self.weights = np.random.rand(grid_dimensions[0], grid_dimensions[1], input_dim)
        self.labels_count = np.zeros((grid_dimensions[0], grid_dimensions[1], 10), dtype=int)

    def _distance(self, a, b):
        return np.linalg.norm(a - b)

    def _find_bmu(self, x):
        min_dist = float('inf')
        bmu_idx = (-1, -1)
        for i in range(self.grid_dimensions[0]):
            for j in range(self.grid_dimensions[1]):
                dist = self._distance(self.weights[i, j], x)
                if dist < min_dist:
                    min_dist = dist
                    bmu_idx = (i, j)
        return bmu_idx

    def _neighborhood_function(self, dist):
        return np.exp(-dist**2 / (2 * self.neighborhood_radius**2))

    def _update_weights(self, x, bmu_idx, learning_rate):
        for i in range(self.grid_dimensions[0]):
            for j in range(self.grid_dimensions[1]):
                dist = self._distance(np.array([i, j]), np.array(bmu_idx))
                if dist <= self.neighborhood_radius:
                    influence = self._neighborhood_function(dist)
                    self.weights[i, j] += learning_rate * influence * (x - self.weights[i, j])

    def train(self, X_train, y_train, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            for x, y in zip(X_train, y_train):
                bmu_idx = self._find_bmu(x)
                self.labels_count[bmu_idx] += np.eye(10, dtype=int)[y]
                self._update_weights(x, bmu_idx, learning_rate)
            learning_rate *= 0.95

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            bmu_idx = self._find_bmu(x)
            y_pred.append(np.argmax(self.labels_count[bmu_idx]))
        return np.array(y_pred)
