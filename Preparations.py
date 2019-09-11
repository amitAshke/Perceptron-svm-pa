import numpy
from scipy import stats

class Preparations :

    def __init__(self, training_data, test_data) :
        self.training_data = training_data
        self.test_data = test_data

    def prepare(self, type):
        self.training_data = self.set_sex_as_bits(['M', 'F', 'I'], self.training_data)
        self.test_data = self.set_sex_as_bits(['M', 'F', 'I'], self.test_data)

        if type == 1 :
            self.training_data = self.min_max_normalization(self.training_data)
            self.test_data = self.min_max_normalization(self.test_data)
        else:
            self.x_train = self.z_score_normalization(self.training_data, 3)
            self.test_data = self.z_score_normalization(self.test_data, 3)

        return self.training_data, self.test_data

    def set_sex_as_bits(self, types_array, data_array):
        collums_added = len(types_array) - 1
        for i in range(collums_added):
            data_array = numpy.c_[numpy.zeros(len(data_array)), data_array]

        for row in range(len(data_array)):
            for col in range(len(types_array)):
                if data_array[row][collums_added] == types_array[col]:
                    data_array[row][collums_added] = float(0)
                    data_array[row][col] = float(1)

        return data_array

    def z_score_normalization(self, data_array, type):
        if type == 0:
            return stats.zscore(data_array.astype(float))
        if type == 1:
            return stats.zscore(data_array.astype(float), ddof=1)
        if type == 2:
            return stats.zscore(data_array.astype(float), axis=1)
        if type == 3:
            return stats.zscore(data_array.astype(float), ddof=1, axis=1)
        if type == 4:
            return stats.mstats.zscore(data_array.astype(float))
        if type == 5:
            return stats.mstats.zscore(data_array.astype(float), ddof=1)
        if type == 6:
            return stats.mstats.zscore(data_array.astype(float), ddof=1, axis=1)
        if type == 7:
            return stats.mstats.zscore(data_array.astype(float), axis=1)

        # features = len(data_array[0])
        #
        # for i in range(features):
        #     data_mean = numpy.mean(data_array[:, i])
        #     data_std = numpy.std(data_array[:, i])
        #     if data_std != 0 and data_std != None and data_mean != None:
        #             data_array[:, i] = (data_array[:, i] - data_mean) / data_std
        # return data_array

    def min_max_normalization(self, data_array):
        for column in range(len(data_array[0])):
            min_arg = float(min(data_array[:, column]))
            max_arg = float(max(data_array[:, column]))

            for row in range(len(data_array)):
                if min_arg == max_arg:
                    data_array[row, column] = 0
                else:
                    data_array[row, column] = (float(data_array[row, column]) - min_arg) / (max_arg - min_arg)

        return data_array