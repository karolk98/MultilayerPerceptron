#!/usr/bin/python3
from enum import Enum
import pandas as pd
import numpy as np

def ReLU(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def dReLU(z):
    return (z > 0) * 1

def dSigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def dTanh(z):
    return 1 / (np.tanh(z) ** 2)

def quad(y1, y2):
    return np.sum((y1-y2)**2)

def dQuad(y1, y2):
    return 2*(y1-y2)


class Perceptron:
    class ProblemType(Enum):
        Classification = 1
        Regression = 2

    def __init__(self, problem_type=ProblemType.Classification, classes=None, hidden_layers=2, hidden_layer_size=None,
                 activation=sigmoid, dActivation=dSigmoid, loss=quad, dLoss=dQuad, final=sigmoid, dFinal=dSigmoid,
                 bias=False,
                 batch_size=100, iterations=20, learning_rate=0.5, momentum=0.5):
        self.problem_type = problem_type
        self.classes = classes if problem_type == self.ProblemType.Classification else 1
        self.hidden_layers = hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.dActivation = dActivation
        self.loss = loss
        self.dLoss = dLoss
        self.final = final
        self.dFinal = dFinal
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
        self.biases = []
        for index, size in enumerate(layer_sizes[1:]):
            size_prev = layer_sizes[index]
            layer = np.random.randn(size, size_prev) * np.sqrt(2 / size_prev)
            self.layers.append(layer)
            if self.bias:
                self.biases.append(np.zeros((size, 1)))

    def train(self):
        self.initialize()
        samples = self.traindata.iloc[:, 0:-1].values
        classes = self.traindata.iloc[:, -1].values

        choice = range(len(samples))
        for a in range(1000):
            index = np.random.choice(choice)
            y = [samples[index]]
            desired_out = np.array([(x == classes[index]) * 1 for x in range(1, self.classes + 1)])

            for idx, layer in enumerate(self.layers):
                output = np.matmul(layer, y[idx].T)
                if idx == len(self.layers)-1:
                    output = self.final(output)
                else:
                    output = self.activation(output)
                y.append(output)

            loss = self.loss(y[-1], desired_out)

            dLoss = self.dLoss(y[-1], desired_out)
            gradients = []
            ygradient = np.multiply(dLoss, self.dFinal(y[-1]))
            gradient = np.outer(ygradient, y[-2])
            gradients.append(gradient)

            for i in range(len(self.layers)-2, -1, -1):
                next_layer = self.layers[i+1]
                ygradient = np.multiply(np.matmul(next_layer.T, ygradient), self.dActivation(y[i+1]))
                gradient = np.outer(ygradient, y[i])
                gradients.append(gradient)

            gradients = gradients[::-1]

            for i in range(len(self.layers)-1, -1, -1):
                self.layers[i] = np.subtract(self.layers[i], gradients[i])

    def test(self, path):
        testdata = pd.read_csv(path)
        samples = testdata.iloc[:, 0:-1].values
        classes = testdata.iloc[:, -1].values
        counter = 0
        for i, sample in enumerate(samples):
            desired_out = classes[i]
            y = sample
            for idx, layer in enumerate(self.layers):
                y = np.matmul(layer, y.T)
                if idx == len(self.layers) - 1:
                    y = self.final(y)
                else:
                    y = self.activation(y)

            predicted = np.argmax(y, axis=0)+1
            if predicted == desired_out:
                counter += 1

        print(counter/len(samples))


perceptron = Perceptron(hidden_layers=2)
perceptron.load(r"C:\Users\Karol\Downloads\project-1-part-1-data\project-1-part-1-data\data.three_gauss.train.10000.csv")
perceptron.train()
perceptron.test(r"C:\Users\Karol\Downloads\project-1-part-1-data\project-1-part-1-data\data.three_gauss.test.10000.csv")
