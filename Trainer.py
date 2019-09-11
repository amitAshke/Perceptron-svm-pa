import numpy
import random
from sklearn.utils import shuffle

class Trainer :

    def __init__(self, training_data, training_labels, classes, lam=0.075, eta=0.02, iterations=100):
        self.training_data = training_data
        self.training_labels = training_labels
        self.lam = lam
        self.eta = eta
        self.iterations = iterations  # how many iterates
        self.perceptron_weights = numpy.zeros((classes, len(training_data[0])))
        self.svm_weights = numpy.zeros((classes, len(training_data[0])))
        self.pa_weights = numpy.zeros((classes, len(training_data[0])))

    def train_all_simul(self):
        for iter in range(self.iterations):
            self.training_data, self.training_labels = shuffle(self.training_data, self.training_labels)

            for x, y in zip(self.training_data, self.training_labels) :
                x = numpy.array(x).astype(float)
                y = int(float(y))

                self.train_perceptron(x, y)
                self.train_svm(x, y)
                self.train_pa(x, y)

            # if self.eta * 0.6 > 0.02 : self.eta *= 0.7
            # else : self.eta = 0.02

            # if self.lam * 0.7 > 0.075: self.eta *= 0.7
            # else: self.lam = 0.075

            # if iter > 0.5 * self.iterations : self.eta = 0.02

            # if iter > 0.5 * self.iterations : self.lam = 0.075

        return self.perceptron_weights, self.svm_weights, self.pa_weights

    def train_perceptron(self, x, y) :
        y_hat = numpy.argmax(numpy.dot(self.perceptron_weights, x))
        if y != y_hat:
            self.perceptron_weights[y, :] = self.perceptron_weights[y, :] + self.eta * x
            self.perceptron_weights[y_hat, :] = self.perceptron_weights[y_hat, :] - self.eta * x

    def train_svm(self, x, y) :
        y_hat = numpy.argmax(numpy.dot(self.svm_weights, x))
        if y != y_hat :
            for iter in range(len(self.svm_weights)) :
                if iter == y :
                    self.svm_weights[y] = (1 - self.eta * self.lam) * self.svm_weights[y] + self.eta * x

                elif iter == y_hat :
                    self.svm_weights[y_hat] = (1 - self.eta * self.lam) * self.svm_weights[y_hat] - self.eta * x

                else :
                    self.svm_weights[iter] *= (1 - self.eta * self.lam)

        else :
            for iter in range(len(self.svm_weights)):
                if iter != y:
                    self.svm_weights[iter] *= (1 - self.eta * self.lam)

    def train_pa(self, x, y) :
        y_hat = numpy.argmax(numpy.dot(self.pa_weights, x))
        if (y != y_hat) :
            x_normalized_power_2 = (2 * (numpy.linalg.norm(x) ** 2))
            if x_normalized_power_2 == 0 : return y_hat
            loss = max(0, (1 - (numpy.dot(self.pa_weights[int(y)], x)) + (numpy.dot(self.pa_weights[int(y_hat)], x))))
            tao = loss / x_normalized_power_2

            self.pa_weights[y] += tao * x
            self.pa_weights[y_hat] -= tao * x
