import numpy as np


class neuralNetwork:
    data = []

    def __init__(self, first_matrix_size, weights_range):
        temp = np.random.uniform(weights_range[0], weights_range[1], size=(first_matrix_size[0], first_matrix_size[1]))
        self.data.append(temp)

    def add_layer(self, n, weights_range):
        if len(weights_range) < 2:
            return -1
        if weights_range[0] > weights_range[1]:
            return -2
        if weights_range[0] < -10 or weights_range[1] < -10:
            return -3
        if weights_range[0] > 10 or weights_range[1] > 10:
            return -4
        previous_shape = []
        previous_shape = self.data[len(self.data) - 1].shape
        temp = np.random.uniform(weights_range[0], weights_range[1], size=(previous_shape[1], n))
        self.data.append(temp)

    def predict(self, input_vector):

        for i in range(len(self.data)):
            input_shape = input_vector.shape
            data_shape = self.data[i].shape

            if data_shape[0] == input_shape[1] or data_shape[1] == input_shape[0]:
                input_vector = self.data[i] * input_vector
            else:
                return -1

        return input_vector

    def load_weights(self, file_name):
        #funkcja po prostu pokazuje że wczytuje jedną macierz i że działa w obie strony
        np.savetxt('test1.txt', self.data[0])
        temp = np.loadtxt('test1.txt')
        print(temp)

    def print_data(self):
        for i in range(len(self.data)):
            print(self.data[i])


if __name__ == '__main__':
    input_vector = np.matrix('0.5; 0.75; 0.1')
    weights_range = [-0.6, 0.6]
    matrix_size = [5, 3]
    myNetwork = neuralNetwork(matrix_size, weights_range)
    # myNetwork.load_weights("test.txt")
    myNetwork.add_layer(5, weights_range)
    out = myNetwork.predict(input_vector)
    print(out)
