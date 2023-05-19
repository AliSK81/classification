import random

import numpy as np
from torch import nn
from torchvision.models import resnet18, resnet34, vgg11_bn, VGG11_BN_Weights, ResNet18_Weights, ResNet34_Weights

from evaluator import evaluate_model
from sgd import SGD
from src.custom_model import CustomModel
from src.data_loader import load_data
from src.dense import Dense
from src.relu import ReLU
from src.sigmoid import Sigmoid
from src.softmax import Softmax
from src.trainer import train_model


class Chromosome:
    def __init__(self, feature_extractor, n_hidden_layers, hidden_layer_neurons, activation_function):
        self.feature_extractor = feature_extractor
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_neurons = hidden_layer_neurons
        self.activation_function = activation_function


def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        feature_extractor = random.choice(['11vgg', '34resnet', '18resnet'])
        n_hidden_layers = random.choice([0, 1, 2])
        hidden_layer_neurons = [random.choice([10, 20, 30]) for _ in range(n_hidden_layers)]
        activation_function = random.choice(['sigmoid', 'relu'])

        chromosome = Chromosome(feature_extractor, n_hidden_layers, hidden_layer_neurons, activation_function)
        population.append(chromosome)
    return population


def selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probabilities = [fitness / total_fitness for fitness in fitness_scores]

    parents = []
    for _ in range(2):
        selected_idx = np.random.choice(len(population), p=selection_probabilities)
        parents.append(population[selected_idx])

    return parents


def crossover(parent1, parent2, crossover_prob=0.5):
    offspring1 = Chromosome(parent1.feature_extractor, parent1.n_hidden_layers, parent1.hidden_layer_neurons.copy(),
                            parent1.activation_function)
    offspring2 = Chromosome(parent2.feature_extractor, parent2.n_hidden_layers, parent2.hidden_layer_neurons.copy(),
                            parent2.activation_function)

    if random.random() < crossover_prob:
        offspring1.feature_extractor, offspring2.feature_extractor = offspring2.feature_extractor, offspring1.feature_extractor

    if random.random() < crossover_prob:
        offspring1.n_hidden_layers, offspring2.n_hidden_layers = offspring2.n_hidden_layers, offspring1.n_hidden_layers
        offspring1.hidden_layer_neurons, offspring2.hidden_layer_neurons = offspring2.hidden_layer_neurons, offspring1.hidden_layer_neurons

    if random.random() < crossover_prob:
        offspring1.activation_function, offspring2.activation_function = offspring2.activation_function, offspring1.activation_function

    return offspring1, offspring2


def mutation(chromosome):
    mutation_choice = random.randint(0, 3)
    if mutation_choice == 0:
        chromosome.feature_extractor = random.choice(['11vgg', '34resnet', '18resnet'])
    elif mutation_choice == 1:
        chromosome.n_hidden_layers = random.choice([0, 1, 2])
        chromosome.hidden_layer_neurons = [random.choice([10, 20, 30]) for _ in range(chromosome.n_hidden_layers)]
    elif mutation_choice == 2:
        for i in range(chromosome.n_hidden_layers):
            chromosome.hidden_layer_neurons[i] = random.choice([10, 20, 30])
    else:
        chromosome.activation_function = random.choice(['sigmoid', 'relu'])
    return chromosome


def run_evolutionary_algorithm(population_size, n_generations, n_executions, train_loader, test_loader):
    population = initialize_population(population_size)

    for generation in range(n_generations):
        fitness_scores = [
            np.mean([evaluate_fitness(chromosome, train_loader, test_loader) for _ in range(n_executions)])
            for chromosome in population
        ]

        for idx, chromosome in enumerate(population):
            log(chromosome, generation, idx, fitness_scores[idx])

        new_population = []
        while len(new_population) < population_size:
            parents = selection(population, fitness_scores)
            offspring1, offspring2 = crossover(parents[0], parents[1])

            offspring1 = mutation(offspring1)
            offspring2 = mutation(offspring2)

            new_population.extend([offspring1, offspring2])

        population = new_population

    fitness_scores = [
        np.mean([evaluate_fitness(chromosome, train_loader, test_loader) for _ in range(n_executions)])
        for chromosome in population
    ]

    best_chromosome_idx = np.argmax(fitness_scores)
    return population[best_chromosome_idx]


def build_and_train(chromosome, train_loader):
    n_classes = len(train_loader.dataset.classes)

    if chromosome.feature_extractor == '18resnet':
        feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif chromosome.feature_extractor == '34resnet':
        feature_extractor = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    else:
        feature_extractor = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)

    if "resnet" in chromosome.feature_extractor:
        n_features = feature_extractor.fc.in_features
        feature_extractor.fc = nn.Identity()
    elif "vgg" in chromosome.feature_extractor:
        n_features = feature_extractor.classifier[-1].in_features
        feature_extractor.classifier = nn.Sequential(*list(feature_extractor.classifier)[:-1])

    layers = []
    n_input = n_features
    for n_neurons in chromosome.hidden_layer_neurons:
        layers.append(Dense(n_input, n_neurons))
        if chromosome.activation_function == 'relu':
            layers.append(ReLU())
        else:
            layers.append(Sigmoid())
        n_input = n_neurons
    layers.append(Dense(n_input, n_classes))
    layers.append(Softmax())

    model = CustomModel(layers=layers)
    optimizer = SGD(learning_rate=0.001)
    epochs = 5
    train_model(model, optimizer, feature_extractor, train_loader, epochs, n_classes)

    return model, feature_extractor


def evaluate_fitness(chromosome, train_loader, test_loader):
    model, feature_extractor = build_and_train(chromosome, train_loader)
    accuracy = evaluate_model(model, feature_extractor, test_loader, "testing")
    return accuracy


def main():
    population_size = 10
    n_generations = 10
    n_executions = 5
    train_loader, test_loader = load_data(batch_size=100)
    best_chromosome = run_evolutionary_algorithm(population_size, n_generations, n_executions, train_loader,
                                                 test_loader)
    model, feature_extractor = build_and_train(best_chromosome, train_loader)
    evaluate_model(model, feature_extractor, train_loader, "training")
    evaluate_model(model, feature_extractor, test_loader, "testing")


def log(chromosome, generation, execution, accuracy):
    print(f"\nGeneration: {generation}, Execution: {execution}")
    print(f"Feature Extractor: {chromosome.feature_extractor}")
    print(f"Number of Hidden Layers: {chromosome.n_hidden_layers}")
    print(f"Hidden Layer Neurons: {chromosome.hidden_layer_neurons}")
    print(f"Activation Function: {chromosome.activation_function}")
    print(f"Accuracy: {accuracy}\n")


if __name__ == '__main__':
    main()
