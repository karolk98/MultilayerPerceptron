#!/usr/bin/python3
from enum import Enum
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.utils import shuffle
from sklearn.metrics import log_loss as CE
from multiprocessing.dummy import Pool as ThreadPool

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import matplotlib
import MnistReader
import time
import os


def Identity(x):
    return x


def dIdentity(x):
    return 1


def ReLU(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def tanh(z):
    return np.tanh(z)

def softmax(z):
    return sp.special.softmax(z, axis=0)

def dReLU(z):
    return (z > 0) * 1


def dSigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def dTanh(z):
    return 1 - (np.tanh(z) ** 2)


def MSE(y1, y2):
    return np.sum((y1 - y2) ** 2) / len(y1)


def dMSE(y1, y2):
    return 2 * (y1 - y2)


def MAE(y1, y2):
    return np.sum(np.abs(y1 - y2)) / len(y1)


def dMAE(y1, y2):
    y = y1 - y2
    for i in range(len(y)):
        y[i] = 1 if y[i] >= 0 else -1
    return y


class Perceptron:
    class ProblemType(Enum):
        Classification = 1
        Regression = 2

    def __init__(self, problem_type=ProblemType.Classification, classes=None, hidden_layers=[2, 2],
                 activation=sigmoid, dActivation=dSigmoid, loss=MSE, dLoss=dMSE, final=sigmoid, dFinal=dSigmoid,
                 SM_CE=False, bias=False,
                 batch_size=1000, epochs=10, learning_rate=0.001, momentum=0.1):
        self.problem_type = problem_type
        self.classes = classes if problem_type == self.ProblemType.Classification else 1
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dActivation = dActivation
        self.loss = loss
        self.dLoss = dLoss
        self.final = final if problem_type == self.ProblemType.Classification else Identity
        self.dFinal = dFinal if problem_type == self.ProblemType.Classification else dIdentity
        if SM_CE:
            self.final = softmax
        self.SM_CE = SM_CE
        self.bias = bias
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.testdata = None
        self.traindata = None

    def load(self, data):
        self.traindata = data

    def load_testdata(self, data):
        self.testdata = data

    def initialize(self):
        self.counter = 0
        self.classes = self.classes if self.classes is not None else len(
            self.traindata[self.traindata.columns[-1]].unique())
        self.data_dim = len(self.traindata.columns) - 1
        layer_sizes = [self.data_dim]
        layer_sizes.extend(self.hidden_layers)
        layer_sizes.append(self.classes)
        self.layers = []
        self.biases = []
        for index, size in enumerate(layer_sizes[1:]):
            size_prev = layer_sizes[index]
            layer = np.random.randn(size, size_prev)
            self.layers.append(layer)
            if self.bias:
                self.biases.append(np.zeros(size))

    def train(self, render_step=None):
        self.initialize()
        samples = self.traindata.iloc[:, 0:-1].values
        classes = self.traindata.iloc[:, -1].values
        gradients = []
        bias_gradients = []
        losses = []
        err_train = []
        err_test = []
        for epoch in range(0, self.epochs):
            print(f"epoch {epoch}")
            batch_start = 0
            while batch_start < len(samples):
                # print(f"samples {batch_start}/{len(samples)}")
                batch_end = min(batch_start + self.batch_size, len(samples))

                batch = samples[batch_start:batch_end].T
                desired = np.zeros((self.classes, batch.shape[1]))
                desired[classes[batch_start:batch_end]-1, np.arange(batch.shape[1])] = 1
                y, activated, _ = self.forward(batch)

                new_gradients, new_bias_gradients, loss = self.gradient(y, activated, desired)

                losses.append(loss/(batch_end-batch_start))

                if len(gradients) == 0:
                    gradients = new_gradients
                    bias_gradients = new_bias_gradients
                else:
                    gradients = [np.add(gradients[idx] * self.momentum, new_gradients[idx]) for
                                 idx in range(len(new_gradients))]
                    bias_gradients = [np.add(bias_gradients[idx] * self.momentum, new_bias_gradients[idx]) for
                                 idx in range(len(new_bias_gradients))]

                should_render = self.initialize_plot(render_step, epoch + 1, int(batch_start / self.batch_size + 1))
                self.apply_gradient(gradients, bias_gradients, should_render)

                batch_start = batch_end

            if self.problem_type == self.ProblemType.Classification and self.testdata is not None:
                etr = self.test_classification(self.traindata)
                err_train.append(etr)
                ete = self.test_classification(self.testdata)
                err_test.append(ete)

        return losses, err_train, err_test

    def forward(self, batch):
        y = [batch]
        output = batch
        activated = [batch]
        for idx, layer in enumerate(self.layers):
            output = np.matmul(layer, output)
            if self.bias:
                output = output + self.biases[idx][:, None]
            y.append(output)
            if idx == len(self.layers) - 1:
                output = self.final(output)
            else:
                output = self.activation(output)
            activated.append(output)
            result = activated[-1]
        return np.array(y, dtype="object"), np.array(activated, dtype="object"), result

    def gradient(self, ys, activated, desired):
        loss = CE(desired, activated[-1]) if self.SM_CE else self.loss(activated[-1], desired)
        gradients = []
        bias_gradients = []
        ygradient = activated[-1] - desired if self.SM_CE else np.multiply(self.dLoss(activated[-1], desired),
                                                                           self.dFinal(ys[-1]))
        if self.bias:
            bias_gradients.append(np.sum(ygradient, axis=1))
        gradient = np.matmul(ygradient, activated[-2].T)
        gradients.append(gradient)

        for i in range(len(self.layers) - 2, -1, -1):
            next_layer = self.layers[i + 1]
            ygradient = np.multiply(np.matmul(next_layer.T, ygradient), self.dActivation(ys[i + 1]))
            if self.bias:
                bias_gradients.append(np.sum(ygradient, axis=1))
            gradient = np.matmul(ygradient, activated[i].T)
            gradients.append(gradient)

        bias_gradients = bias_gradients[::-1]
        gradients = gradients[::-1]
        return gradients, bias_gradients, loss

    def initialize_plot(self, render_step, epoch, batch):
        if not render_step:
            return False
        should_render = self.counter == 0
        if should_render:
            self.fig = plt.figure()
            self.fig.suptitle(f"Epoch: {epoch}  Batch: {batch}")
            ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers) + 1, 1)
            ax.axis('off')
            ax.text(0.5, 0.5, "Weights")
            ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers) + 1, len(self.layers) + 2)
            ax.axis('off')
            ax.text(0.5, 0.5, "Gradients")
            if self.bias:
                ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers) + 1, 2 * len(self.layers) + 3)
                ax.axis('off')
                ax.text(0.5, 0.5, "Bias")
                ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers) + 1, 3 * len(self.layers) + 4)
                ax.axis('off')
                ax.text(0.5, 0.5, "Gradient")
        self.counter = (self.counter + 1) % render_step
        return should_render

    def apply_gradient(self, gradients, bias_gradients, interactive):
        for i in range(len(self.layers)):
            if self.bias:
                self.biases[i] = np.subtract(self.biases[i], self.learning_rate * bias_gradients[i])
            self.layers[i] = np.subtract(self.layers[i], self.learning_rate * gradients[i])
            if interactive:
                ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers) + 1, i + 2)
                plt.colorbar(ax.matshow(self.layers[i], cmap=plt.cm.Blues))
                ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers) + 1, len(self.layers) + i + 3)
                if self.bias:
                    plt.colorbar(ax.matshow(gradients[i], cmap=plt.cm.Blues))
                    ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers) + 1, 2 * len(self.layers) + i + 4)
                    plt.colorbar(ax.matshow(self.biases[i].reshape(len(self.biases[i]), 1), cmap=plt.cm.Blues))
                    ax.get_xaxis().set_visible(False)
                    ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers) + 1, 3 * len(self.layers) + i + 5)
                    plt.colorbar(
                        ax.matshow(bias_gradients[i].reshape(len(bias_gradients[i]), 1), cmap=plt.cm.Blues))
                    ax.get_xaxis().set_visible(False)
                else:
                    plt.colorbar(ax.matshow(gradients[i], cmap=plt.cm.Blues))
        if interactive:
            plt.show()
            input()

    def test_classification(self, testdata):
        samples = testdata.iloc[:, 0:-1].values
        classes = testdata.iloc[:, -1].values-1
        y, act, result = self.forward(samples.T)
        correct = np.sum(np.argmax(result, axis=0) == classes)

        return correct / len(samples)


def draw_classification(network, path):
    samples = pd.read_csv(path)
    samples = shuffle(samples)
    x = samples.iloc[:, 0]
    y = samples.iloc[:, 1]
    classes = samples.iloc[:, 2]
    colors = ['red', 'blue', 'green']
    xlin = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
    ylin = np.linspace(y.min() - 0.1, y.max() + 0.1, 200)
    xm, ym = np.meshgrid(xlin, ylin)
    res = np.zeros((xlin.shape[0], ylin.shape[0]), dtype=int)
    for i, xi in enumerate(xlin[:-1]):
        for j, yj in enumerate(ylin[:-1]):
            _, _, ans = network.forward([xi, yj])
            res[j, i] = np.argmax(ans, axis=0)
    plt.pcolor(xm, ym, res,
               shading='auto',
               cmap=matplotlib.colors.ListedColormap(colors),
               alpha=0.4, snap=True)

    plt.scatter(x, y,
                c=classes,
                s=0.2,
                cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()


def draw_regression2d(network, x, func):
    args = x
    values = [func(a) for a in args]
    predicted_values = []
    for i, sample in enumerate(args):
        _, _, result = network.forward([sample])
        predicted_values.append(result)

    plt.scatter(args, values, label="expectation")
    plt.scatter(args, predicted_values, label="prediction")
    plt.legend(loc="best")
    plt.show()


def draw_regression3d(network, x, y, func):
    xm, ym = np.meshgrid(x, y)
    res = np.zeros((x.shape[0], y.shape[0]), dtype=float)
    predicted_values = np.zeros((x.shape[0], y.shape[0]), dtype=float)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            res[j, i] = func(xi, yj)
            _, _, pred = network.forward([xi, yj])
            predicted_values[j, i] = pred

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(25, 80)
    s1 = ax.plot_surface(xm, ym, res,
                         label="expectation",
                         color='red',
                         edgecolor='none')

    s2 = ax.plot_surface(xm, ym, predicted_values,
                         label="prediction",
                         color='blue',
                         edgecolor='none')
    red_patch = mpatches.Patch(color='red', label='expectation')
    blue_patch = mpatches.Patch(color='blue', label="prediction")
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()


def plot_errors(losses, title="Error depending on a pass number"):
    plt.plot(range(len(losses)), losses, label="one pass")
    mean_losses = []
    for i in range(0, len(losses), 100):
        vv = min(len(losses), i + 100)
        mean_losses.append(sum(losses[i:vv]) / 100)
    xes = range(len(mean_losses))
    xes = [v * 100 for v in xes]
    plt.plot(xes, mean_losses, label="mean of 100 passes")
    plt.xlabel("pass")
    plt.ylabel("error")
    plt.title(title)
    plt.legend()
    plt.show()

    print(f'Last mean error: {mean_losses[-1]}')


def plot_accuracy(epochs, train_accuracy, test_accuracy):
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(range(1, epochs + 1), train_accuracy, label="train data")
    ax.plot(range(1, epochs + 1), test_accuracy, label="test data")
    ax.set_ylabel("epoch")
    ax.set_xlabel("accuracy")
    ax.set_title("Accuracy on test and train data")
    ax.legend()
    plt.show()


def run_tests(tests, testdata, traindata):
    for test in tests:
        path = time.strftime("%Y-%m-%d_%H-%M-%S")
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)
        with open(os.path.join(path, 'description.txt'), 'w') as f:
            print(test, file=f)

        mlp = Perceptron(problem_type=test["problem_type"],
                         hidden_layers=test["hidden_layers"],
                         activation=test["activation"],
                         dActivation=test["dActivation"],
                         SM_CE=test["SM_CE"],
                         batch_size=test["batch_size"],
                         learning_rate=test["learning_rate"],
                         momentum=test["momentum"],
                         epochs=test["epochs"],
                         bias=test["bias"])
        mlp.load(traindata)
        mlp.load_testdata(testdata)
        losses, train_accuracy, test_accuracy = mlp.train()

        np.savetxt(os.path.join(path, "losses.csv"), np.asarray(losses), delimiter=",")
        np.savetxt(os.path.join(path, "train_accuracy.csv"), np.asarray(train_accuracy), delimiter=",")
        np.savetxt(os.path.join(path, "test_accuracy.csv"), np.asarray(test_accuracy), delimiter=",")

        print(f'train accuracy: {train_accuracy}')
        print(f'test accuracy: {test_accuracy}')

        plot_accuracy(test["epochs"], train_accuracy, test_accuracy)

        plot_errors(losses)


if __name__ == "__main__":
    np.random.seed(1)

    tests = [
        {"problem_type": Perceptron.ProblemType.Classification,
         "hidden_layers": [128, 16],
         "activation": sigmoid,
         "dActivation": dSigmoid,
         "SM_CE": True,
         "batch_size": 100,
         "learning_rate": 1e-5,
         "momentum": 0.9,
         "epochs": 1000,
         "bias": True
         },
    ]

    run_tests(tests,
              traindata=MnistReader.load_data("data\\MNIST\\raw\\train-images-idx3-ubyte",
                                              "data\\MNIST\\raw\\train-labels-idx1-ubyte"),
              testdata=MnistReader.load_data("data\\MNIST\\raw\\t10k-images-idx3-ubyte",
                                             "data\\MNIST\\raw\\t10k-labels-idx1-ubyte"))
