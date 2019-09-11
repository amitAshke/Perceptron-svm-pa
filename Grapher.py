import numpy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import Trainer

class Grapher :
    def __init__(self, training_data, training_labels, test_data, test_label, classes, initial_lambda=0.075, initial_eta=0.02, iterations=200):
        self.training_data = training_data
        self.training_labels = training_labels
        self.test_data = test_data
        self.test_label = test_label
        self.initial_lambda = initial_lambda
        self.initial_eta = initial_eta
        self.iterations = iterations  # how many iterates
        self.classes = classes

    def perceptron_graph(self):
        array_eta = []
        array_iter = []
        array_preceptron_perc = []
        array_svm_perc = []
        array_pa_perc = []

        perceptron_weights = numpy.zeros((self.classes, len(self.training_data[0])))
        svm_weights = numpy.zeros((self.classes, len(self.training_data[0])))
        pa_weights = numpy.zeros((self.classes, len(self.training_data[0])))

        for iter in range(self.iterations):
            # self.training_data, self.training_labels = shuffle(self.training_data, self.training_labels)

            array_eta.append(self.initial_eta)
            array_iter.append(iter)

            # train perceptron
            for x, y in zip(self.training_data, self.training_labels) :
                x = numpy.array(x).astype(float)
                y = int(float(y))

                self.train_perceptron(x, y, perceptron_weights)
                self.train_svm(x, y, svm_weights)
                self.train_pa(x, y, pa_weights)

            # measure success rate
            perceptron_counter = 0
            svm_counter = 0
            pa_counter = 0
            for t in range(len(self.test_data)):
                x = numpy.array(self.test_data[t]).astype(float)
                perceptron_counter += self.test_algorithm(self.test_label[t], x, perceptron_weights)
                svm_counter += self.test_algorithm(self.test_label[t], x, svm_weights)
                pa_counter += self.test_algorithm(self.test_label[t], x, pa_weights)

            array_preceptron_perc.append(perceptron_counter / len(self.test_data))
            array_svm_perc.append(svm_counter / len(self.test_data))
            array_pa_perc.append(pa_counter / len(self.test_data))

            # self.initial_eta += 0.001
            # perceptron_weights = numpy.zeros((self.classes, len(self.training_data[0])))

        self.plot_2d_grid(array_iter, array_preceptron_perc, "perceptron - precentage of success for iteration","iteration", "precentage of success")
        # self.plot_2d_grid(array_iter, array_svm_perc, "svm - precentage of success for iteration","iteration", "precentage of success")
        # self.plot_2d_grid(array_iter, array_pa_perc, "pa - precentage of success for iteration","iteration", "precentage of success")

        # self.plot_2d_grid(array_eta, array_preceptron_perc, "perceptron - precentage of success for eta","eta", "precentage of success")
        # self.plot_2d_grid(array_eta, array_svm_perc, "svm - precentage of success for eta","eta", "precentage of success")


    def train_perceptron(self, x, y, perceptron_weights):
        y_hat = numpy.argmax(numpy.dot(perceptron_weights, x))
        if y != y_hat:
            perceptron_weights[y, :] = perceptron_weights[y, :] + self.initial_eta * x
            perceptron_weights[y_hat, :] = perceptron_weights[y_hat, :] - self.initial_eta * x

    def train_svm(self, x, y, svm_weights) :
        y_hat = numpy.argmax(numpy.dot(svm_weights, x))
        if y != y_hat :
            for iter in range(len(svm_weights)) :
                if iter == y :
                    svm_weights[y] = (1 - self.initial_eta * self.initial_lambda) * svm_weights[y] + self.initial_eta * x

                elif iter == y_hat :
                    svm_weights[y_hat] = (1 - self.initial_eta * self.initial_lambda) * svm_weights[y_hat] - self.initial_eta * x

                else :
                    svm_weights[iter] *= (1 - self.initial_eta * self.initial_lambda)

        else :
            for iter in range(len(svm_weights)):
                if iter != y:
                    svm_weights[iter] *= (1 - self.initial_eta * self.initial_lambda)

    def train_pa(self, x, y, pa_weights) :
        y_hat = numpy.argmax(numpy.dot(pa_weights, x))
        if (y != y_hat) :
            x_normalized_power_2 = (2 * (numpy.linalg.norm(x) ** 2))
            if x_normalized_power_2 == 0 : return y_hat
            loss = max(0, (1 - (numpy.dot(pa_weights[int(y)], x)) + (numpy.dot(pa_weights[int(y_hat)], x))))
            tao = loss / x_normalized_power_2

            pa_weights[y] += tao * x
            pa_weights[y_hat] -= tao * x

    def test_algorithm(self, y, x, weights):
        y_hat = numpy.argmax(numpy.dot(weights, x))
        if y == y_hat:
            return 1
        return 0

    def plot_2d_grid(self, x, y, title, x_label, y_label):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()