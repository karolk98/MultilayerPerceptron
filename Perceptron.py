#!/usr/bin/python3
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib


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


def dReLU(z):
    return (z > 0) * 1


def dSigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def dTanh(z):
    return 1 - (np.tanh(z) ** 2)


def quad(y1, y2):
    return np.sum((y1 - y2) ** 2)


def dQuad(y1, y2):
    return 2 * (y1 - y2)


class Perceptron:
    class ProblemType(Enum):
        Classification = 1
        Regression = 2

    def __init__(self, problem_type=ProblemType.Classification, classes=None, hidden_layers=[2, 2],
                 activation=sigmoid, dActivation=dSigmoid, loss=quad, dLoss=dQuad, final=sigmoid, dFinal=dSigmoid,
                 bias=False,
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
        self.bias = bias
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum

    def load(self, path):
        self.traindata = pd.read_csv(path)
        self.traindata = shuffle(self.traindata)

    def initialize(self):
        self.counter = 0
        self.classes = self.classes if self.classes is not None else len(self.traindata['cls'].unique())
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
        losses = []
        for epoch in range(0, self.epochs):
            batch_start = 0
            while batch_start < len(samples):
                batch_end = min(batch_start + self.batch_size, len(samples))

                batch_size = batch_end - batch_start
                new_gradients = []

                for m in range(batch_start, batch_end):
                    y, activated, _ = self.forward(samples[m])
                    desired_out = np.zeros(self.classes)

                    if self.problem_type == self.ProblemType.Classification:
                        desired_out[classes[m] - 1] = 1
                    else:
                        desired_out[0] = classes[m]

                    mth_gradients, loss = self.gradient(y, activated, desired_out)
                    losses.append(loss)
                    if m == batch_start:
                        new_gradients = mth_gradients
                    else:
                        new_gradients = [np.add(new_gradients[idx], mth_gradients[idx]) for idx in
                                         range(len(new_gradients))]

                new_gradients = [gradient / batch_size for gradient in new_gradients]
                if len(gradients) == 0:
                    gradients = new_gradients
                else:
                    gradients = [np.add(gradients[idx] * self.momentum, new_gradients[idx] * (1 - self.momentum)) for
                                 idx in range(len(new_gradients))]

                should_render = self.initialize_plot(render_step, epoch+1, int(batch_start / self.batch_size + 1))
                self.apply_gradient(gradients, should_render)

                batch_start = batch_end
        return losses

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
            result = activated[-1]
        return y, activated, result

    def gradient(self, ys, activated, desired):
        loss = self.loss(activated[-1], desired)
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
        return gradients, loss

    def initialize_plot(self, render_step, epoch, batch):
        if not render_step:
            return False
        should_render = self.counter==0
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
        self.counter = (self.counter+1)%render_step
        return should_render


    def apply_gradient(self, gradients, interactive):
        for i in range(len(self.layers)):
            if self.bias:
                self.layers[i] = np.subtract(self.layers[i], self.learning_rate * gradients[i][:, :-1])
                self.biases[i] = np.subtract(self.biases[i], self.learning_rate * gradients[i][:, -1])
            else:
                self.layers[i] = np.subtract(self.layers[i], self.learning_rate * gradients[i])
            if interactive:
                ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers)+1, i + 2)
                plt.colorbar(ax.matshow(self.layers[i], cmap=plt.cm.Blues))
                ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers)+1, len(self.layers) + i + 3)
                if self.bias:
                    plt.colorbar(ax.matshow(gradients[i][:, :-1], cmap=plt.cm.Blues))
                    ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers) + 1, 2*len(self.layers) + i + 4)
                    plt.colorbar(ax.matshow(self.biases[i].reshape(len(self.biases[i]),1), cmap=plt.cm.Blues))
                    ax.get_xaxis().set_visible(False)
                    ax = self.fig.add_subplot(2 + 2 * self.bias, len(self.layers) + 1, 3*len(self.layers) + i + 5)
                    plt.colorbar(ax.matshow(gradients[i][:, -1].reshape(len(gradients[i][:, -1]),1), cmap=plt.cm.Blues))
                    ax.get_xaxis().set_visible(False)
                else:
                    plt.colorbar(ax.matshow(gradients[i], cmap=plt.cm.Blues))
        if interactive:
            plt.show()
            input()

    def test_classification(self, path):
        testdata = pd.read_csv(path)
        samples = testdata.iloc[:, 0:-1].values
        classes = testdata.iloc[:, -1].values
        counter = 0
        for i, sample in enumerate(samples):
            desired_out = classes[i]
            y, act, result = self.forward(sample)

            predicted = np.argmax(result, axis=0) + 1
            if predicted == desired_out:
                counter += 1

        print(f'Success ratio: {counter / len(samples)}')


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
               cmap=matplotlib.colors.ListedColormap(colors),
               alpha=0.4, shading='auto', snap=True)

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


def plot_errors(losses):
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
    plt.title("error change")
    plt.legend()
    plt.show()

    print(f'Last mean error: {mean_losses[-1]}')


def plot_errors_title(losses, title):
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


if __name__ == "__main__":
    np.random.seed(1)
    #
    # mlp = Perceptron(problem_type=Perceptron.ProblemType.Regression,
    #                  hidden_layers=[10],
    #                  activation=sigmoid,
    #                  dActivation=dSigmoid,
    #                  batch_size=3,
    #                  learning_rate=0.001,
    #                  momentum=0.1,
    #                  epochs=1,
    #                  bias=True)
    #
    # mlp.load("data\quadratic2_train.csv")
    # losses = mlp.train()
    # draw_regression2d(mlp, x=np.linspace(-5, 5, 100), func=lambda x: x ** 2-3*x-5)
    # #plot_errors(losses)

    # mlp = Perceptron(problem_type=Perceptron.ProblemType.Regression,
    #                  hidden_layers=[12],
    #                  activation=sigmoid,
    #                  dActivation=dSigmoid,
    #                  batch_size=3,
    #                  learning_rate=0.01,
    #                  momentum=0.1,
    #                  epochs=10,
    #                  bias=True)
    #
    # mlp.load("data\\3d_train.csv")
    # losses = mlp.train()
    # draw_regression3d(mlp,
    #                   x=np.linspace(-5, 5, 50),
    #                   y=np.linspace(-5, 5, 50),
    #                   func=lambda x, y:  x**2+x*y+5*x-1)
    #
    # np.random.seed(1)

    layers = [[],
              [3],
              [3, 3],
              [3, 3, 3],
              [3, 3, 3, 3],
              [3, 3, 3, 3, 3]]
    for ind, layer_type in enumerate(layers):
        mlp = Perceptron(problem_type=Perceptron.ProblemType.Classification,
                         hidden_layers=layer_type,
                         activation=sigmoid,
                         dActivation=dSigmoid,
                         final=sigmoid,
                         dFinal=dSigmoid,
                         batch_size=3,
                         learning_rate=0.05,
                         momentum=0.1,
                         epochs=1,
                         bias=False)
        mlp.load("data\data.three_gauss.train.10000.csv")
        losses = mlp.train(5)
        st = f'{ind} hidden layers' if ind != 1 else f'{ind} hidden layer'
        print(f'{st}, last mean error: {losses[-1]}')
        plot_errors_title(losses, st)