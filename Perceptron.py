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

    def __init__(self, problem_type=ProblemType.Classification, classes=None, hidden_layers=[2, 2],
                 activation=sigmoid, dActivation=dSigmoid, loss=quad, dLoss=dQuad, final=sigmoid, dFinal=dSigmoid,
                 bias=False,
                 batch_size=1000, epochs=10, learning_rate=0.5, momentum=0.5):
        self.problem_type = problem_type
        self.classes = classes if problem_type == self.ProblemType.Classification else 1
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dActivation = dActivation
        self.loss = loss
        self.dLoss = dLoss
        self.final = final
        self.dFinal = dFinal
        self.bias = bias
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum

    def load(self, path):
        self.traindata = pd.read_csv(path)

    def initialize(self):
        self.classes = self.classes if self.classes is not None else len(self.traindata['cls'].unique())
        self.data_dim = len(self.traindata.columns) - 1
        layer_sizes = [self.data_dim]
        layer_sizes.extend(self.hidden_layers)
        layer_sizes.append(self.classes)
        self.layers = []
        self.biases = []
        for index, size in enumerate(layer_sizes[1:]):
            size_prev = layer_sizes[index]
            layer = np.random.randn(size, size_prev) * np.sqrt(2 / size_prev)
            self.layers.append(layer)
            if self.bias:
                self.biases.append(np.zeros(size))

    def train(self):
        self.initialize()
        samples = self.traindata.iloc[:, 0:-1].values
        classes = self.traindata.iloc[:, -1].values

        gradients = []
        for epoch in range(0, self.epochs):
            batch_start = 0
            while batch_start < len(samples):
                batch_end = min(batch_start + self.batch_size, len(samples))

                batch_size = batch_end-batch_start
                new_gradients = []

                for m in range(batch_start, batch_end):
                    y, activated = self.forward(samples[m])
                    desired_out = np.zeros(self.classes)
                    desired_out[classes[m]-1] = 1
                    mth_gradients = self.gradient(y, activated, desired_out)
                    if m == batch_start:
                        new_gradients = mth_gradients
                    else:
                        new_gradients = [np.add(new_gradients[idx], mth_gradients[idx]) for idx in range(len(new_gradients))]

                new_gradients = [gradient/batch_size for gradient in new_gradients]
                if len(gradients) == 0:
                    gradients = new_gradients
                else:
                    gradients = [np.add(gradients[idx]*self.momentum, new_gradients[idx]*(1-self.momentum)) for idx in range(len(new_gradients))]

                self.apply_gradient(gradients)

                batch_start = batch_end

    def forward(self, batch):
        y = [batch]
        output = batch
        activated = [batch]
        for idx, layer in enumerate(self.layers):
            output = np.matmul(layer, output)
            if self.bias:
                output = output + self.biases[idx]
            y.append(output)
            if idx == len(self.layers) - 1:
                output = self.final(output)
            else:
                output = self.activation(output)
            activated.append(output)
        return y, activated

    def gradient(self, ys, activated, desired):
        dLoss = self.dLoss(activated[-1], desired)
        gradients = []
        ygradient = np.multiply(dLoss, self.dFinal(ys[-1]))
        if self.bias:
            gradient = np.outer(ygradient, np.append(activated[-2], 1))
        else:
            gradient = np.outer(ygradient, activated[-2])
        gradients.append(gradient)

        for i in range(len(self.layers) - 2, -1, -1):
            next_layer = self.layers[i + 1]
            ygradient = np.multiply(np.matmul(next_layer.T, ygradient), self.dActivation(ys[i + 1]))
            if self.bias:
                gradient = np.outer(ygradient, np.append(activated[i], 1))
            else:
                gradient = np.outer(ygradient, activated[i])
            gradients.append(gradient)

        gradients = gradients[::-1]
        return gradients

    def apply_gradient(self, gradients):
        for i in range(len(self.layers)):
            if self.bias:
                self.layers[i] = np.subtract(self.layers[i], self.learning_rate * gradients[i][:, :-1])
                self.biases[i] = np.subtract(self.biases[i], self.learning_rate * gradients[i][:, -1])
            else:
                self.layers[i] = np.subtract(self.layers[i], self.learning_rate * gradients[i])

    def test(self, path):
        testdata = pd.read_csv(path)
        samples = testdata.iloc[:, 0:-1].values
        classes = testdata.iloc[:, -1].values
        counter = 0
        for i, sample in enumerate(samples):
            desired_out = classes[i]
            y, _ = self.forward(sample)

            predicted = np.argmax(y[-1], axis=0)+1
            if predicted == desired_out:
                counter += 1

        print(counter/len(samples))


if __name__ == "__main__":
    np.random.seed(676)
    perceptron = Perceptron(hidden_layers=[2, 3], batch_size=10, epochs=1, bias=True)
    perceptron.load(r"C:\Users\Potato\Downloads\project-1-part-1-data\project-1-part-1-data\data.simple.train.10000.csv")
    perceptron.train()
    perceptron.test(r"C:\Users\Potato\Downloads\project-1-part-1-data\project-1-part-1-data\data.simple.test.10000.csv")
