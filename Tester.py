import numpy

class Tester :
    def __init__(self, test_data, perceptron_weights, svm_weights, pa_weights) :
        self.test_data = test_data
        self.perceptron_weights = perceptron_weights
        self.svm_weights = svm_weights
        self.pa_weights = pa_weights

    def test(self) :
        for iter in range(len(self.test_data)) :
            single_data = numpy.array(self.test_data[iter]).astype(float)
            perceptron_prediction = numpy.argmax(numpy.dot(self.perceptron_weights, single_data))
            svm_prediction = numpy.argmax(numpy.dot(self.svm_weights, single_data))
            pa_prediction = numpy.argmax(numpy.dot(self.pa_weights, single_data))
            print("perceptron: {}, svm: {}, pa: {}".format(perceptron_prediction, svm_prediction, pa_prediction))

    def test_algorithm(self, y, x, weights):
        y_hat = numpy.argmax(numpy.dot(weights, x))
        if y == y_hat:
            return 1
        return 0

    def calculate_statistics(self, test_label) :
        perceptron_counter = 0
        svm_counter = 0
        pa_counter = 0
        for t in range(len(self.test_data)):
            x = numpy.array(self.test_data[t]).astype(float)
            perceptron_counter += self.test_algorithm(test_label[t], x, self.perceptron_weights)
            svm_counter += self.test_algorithm(test_label[t], x, self.svm_weights)
            pa_counter += self.test_algorithm(test_label[t], x, self.pa_weights)
        perceptron_success_rate = perceptron_counter / len(self.test_data)
        svm_success_rate = svm_counter / len(self.test_data)
        pa_success_rate =  pa_counter / len(self.test_data)
        return perceptron_success_rate, svm_success_rate, pa_success_rate