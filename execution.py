import numpy
import sys

import Grapher
import Preparations
import Trainer
import Tester

if __name__ == '__main__' :

    if len(sys.argv) < 4 :
        print("Error: missing files")
        exit(-1)

    classes = 3

    training_data = numpy.genfromtxt(sys.argv[1], delimiter=',', dtype="|U5")
    training_labels = numpy.genfromtxt(sys.argv[2], delimiter=',')
    test_data = numpy.genfromtxt(sys.argv[3], delimiter=',', dtype="|U5")

    training_data, test_data = Preparations.Preparations(training_data, test_data).prepare(1)

    perceptron_weights, svm_weights, pa_weights = Trainer.Trainer(training_data, training_labels, classes).train_all_simul()

    tester = Tester.Tester(test_data, perceptron_weights, svm_weights, pa_weights)

    tester.test()

    if len(sys.argv) == 5 and True:  # debug mode
        perceptron_success_rate, svm_success_rate, pa_success_rate = tester.calculate_statistics(numpy.genfromtxt(sys.argv[4], delimiter=','))
        print("succeeds rate: per: {}, svm:{}, pa: {}".format(perceptron_success_rate, svm_success_rate, pa_success_rate))
        Grapher.Grapher(training_data, training_labels, test_data, numpy.genfromtxt(sys.argv[4]), classes).perceptron_graph()
    else:  # testing mode
        tester.test()
