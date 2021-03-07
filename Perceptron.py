#!/usr/bin/python3
from enum import Enum
import pandas as pd
from numpy import *


class Perceptron:
    class ProblemType(Enum):
        Classification = 1
        Regression = 2

    def __init__(self, problem_type=ProblemType.Classification, classes=None, hidden_layers=2, hidden_layer_size=None,
                 activation_function=None,
                 bias=True,
                 batch_size=100, iterations=20, learning_rate=0.5, momentum=0.5):
        self.problem_type = problem_type
        self.classes = classes if problem_type == self.ProblemType.Classification else 1
        self.hidden_layers = hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation_function = activation_function
        self.bias = bias
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.momentum = momentum

    def load(self, path):
        self.traindata = pd.read_csv(path)
        print(self.traindata[0:10])

    def initialize(self):
        self.classes = self.classes if self.classes is not None else len(self.traindata['cls'].unique())
        self.data_dim = len(self.traindata.columns) - 1
        self.hidden_layer_size = self.hidden_layer_size if self.hidden_layer_size is not None else int(
            (self.data_dim + self.classes) / 2)
        layer_sizes = [self.data_dim]
        layer_sizes.extend([self.hidden_layer_size for x in range(self.hidden_layers)])
        layer_sizes.append(self.classes)
        self.layers = []
        for index, size in enumerate(layer_sizes[1:]):
            size_prev = layer_sizes[index]
            layer = array([[np.random.randn(size, size_prev) * np.sqrt(2 / size_prev)] * size_prev] * size)
            self.layers.append(layer)
        if self.bias:
            self.biases = [0]*(self.hidden_layers+1)

    def learn(self):
        pass


perceptron = Perceptron()
perceptron.load(r"C:\Users\Karol\Downloads\project-1-part-1-data\project-1-part-1-data\data.simple.train.10000.csv")
perceptron.initialize()
