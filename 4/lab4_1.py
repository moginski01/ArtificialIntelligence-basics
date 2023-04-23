import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import time
import random

class neuralNetwork:
    data = []
    input_data = []
    error = 0
    labels_dec = []

    def __init__(self, first_matrix_size, weights_range):
      
        temp = np.random.uniform(weights_range[0], weights_range[1], size=(first_matrix_size[0], first_matrix_size[1]))
        self.data.append(temp)

    def add_layer(self, weights):
        if len(self.data) == 0:
            self.data.append(weights)

    def ReLU_deriv(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def ReLU(self, x):
        return np.maximum(0, x)

    def add_layer_rand(self, size, weights_range):
        temp = np.asmatrix(np.random.uniform(weights_range[0], weights_range[1], size=(size[0], size[1])))
        # new_shape = temp.shape
        self.data.append(temp)

    def clear_data(self):
        self.data.clear()

    def predict(self, input_vector, layer_number):
        return np.asmatrix(self.data[layer_number]) * np.asmatrix(input_vector)

    def update_weights(self, expected, alpha, cycles, how_many_img, percent_to_erase):

        correct_counter = 0
        # print(len(self.data))
        out = []
        res = []
        temp = 0
        hidden_number_of_elements = self.data[0].shape[0] * self.data[0].shape[1]
        # na razie na 1 ustawione
        for b in range(cycles):
            correct_counter = 0
            # a numer serri
            # pętla dla serii
            for a in range(how_many_img):
                out = []
                # pętla dla wag hidden -> output
                for i in range(len(self.data)):
                    if i == 0:
                        # print(self.input_data[0][a,:,0])

                        temp = self.predict(np.asmatrix(self.input_data[0][a]), 0)
                        temp = self.ReLU(temp)
                        out.append(temp)
                    else:
                        dropout_mask = np.random.randint(2,size=out[-1].shape)
                        out[-1] = np.multiply(out[-1],dropout_mask)*2
                        #razy 2 mnozymy pozostale wagi ponieważ o 50% wag zerujemy
                        temp = self.predict(np.asmatrix(out[-1]), 1)

                        out.append(temp)

                        layer_output_delta = (2 / out[-1].shape[0]) * (out[-1] - expected[a])

                        layer_hidden_1_delta = self.data[i].T * layer_output_delta

                        temp_vec_function = np.vectorize(self.ReLU_deriv)
                        layer_hidden_1_delta = np.multiply(layer_hidden_1_delta, temp_vec_function(out[-2]))
                        layer_hidden_1_delta = np.multiply(layer_hidden_1_delta,dropout_mask)

                        layer_output_weight_delta = np.asmatrix(layer_output_delta) * np.asmatrix(out[-2]).T
                        layer_hidden_1_weight_delta = np.asmatrix(layer_hidden_1_delta) * np.asmatrix(self.input_data[0][a]).T

                        maxRow = out[-1][0, 0]
                        maxIndex = 0
                                          
                        for j in range(10):
                            if out[-1][j, 0] >= maxRow:
                                maxRow = out[-1][j, 0]
                                maxIndex = j
                        if expected[a][maxIndex, 0] == 1:
                            correct_counter += 1
                        else:
                            res.append([out[-1], expected[a], a])
                        self.data[0] = np.asmatrix(self.data[0]) - np.asmatrix(alpha * layer_hidden_1_weight_delta)
                        self.data[1] = np.asmatrix(self.data[1]) - np.asmatrix(alpha * layer_output_weight_delta)

            print("Procent poprawnych: " + "numer serii:  " + str(b))
            print(correct_counter/how_many_img*100)
           

        return correct_counter, res

    def smart_neural_network(self, expected):
        
        correct_counter = 0
        # print(len(self.data))
        out = []
        res = []
        temp = 0
        # na razie na 1 ustawione
        correct_counter = 0
        # a numer serri
        # pętla dla serii
        for a in range(self.input_data[0].shape[0]):
            out = []
            # pętla dla wag hidden -> output
            for i in range(len(self.data)):
                if i == 0:
                    # print(self.input_data[0][a,:,0])
                    temp = self.predict(np.asmatrix(self.input_data[0][a]), 0)
                    temp = self.ReLU(temp)
                    out.append(temp)
                else:
                    temp = self.predict(np.asmatrix(out[-1]), 1)
                    out.append(temp)                       

                    maxRow = out[-1][0, 0]
                    maxIndex = 0
         

                    for j in range(10):
                        if out[-1][j, 0] >= maxRow:
                            maxRow = out[-1][j, 0]
                            maxIndex = j
                    if expected[a][maxIndex, 0] == 1:
                        correct_counter += 1

        print("Procent poprawnych: ")
        print(correct_counter/10000*100)

        return correct_counter, res


    def save_weights_og(self):
        np.savetxt("wagi1og.txt",self.data[0])
        np.savetxt("wagi2og.txt",self.data[1])

    def save_weights(self):
        np.savetxt("wagi1.txt",self.data[0])
        np.savetxt("wagi2.txt",self.data[1])

    def load_weights(self, file_name):
        # np.savetxt('test1.txt', self.data[0])
        weights1 = np.loadtxt('weights1.txt')
        weights2 = np.loadtxt('weights2.txt')
        self.data.append(weights1)
        self.data.append(weights2)
        # print(temp)

    def load_multiple_weights(self, file_name):
        print("funload")
        temp = np.loadtxt(file_name)
        temp_matrix = np.zeros((4, 3))
        temp_matrix = np.asmatrix(temp_matrix)
        how_many_matrixes = float(temp.shape[0]) / float(temp_matrix.shape[1])

        for i in range(int(how_many_matrixes)):
            temp_matrix = temp[(i * 3):((i + 1) * 3), :]
            self.data.append(temp_matrix)

    def load_multiple_inputs(self, file_name):
        temp = np.loadtxt(file_name)
        temp_matrix = np.zeros((1, 4))
        temp_matrix = np.asmatrix(temp_matrix)
        how_many_matrixes = float(temp.shape[0])
        # print(how_many_matrixes)
        for i in range(int(how_many_matrixes)):
            temp_matrix = temp[i, :]
            # print(temp_matrix)
            self.input_data.append(temp_matrix)

    def load_mnist(self):
        file1 = 'train-labels.idx1-ubyte'
        file2 = 'train-images.idx3-ubyte'
        labels = idx2numpy.convert_from_file(file1)
        images = idx2numpy.convert_from_file(file2)
        images = images/255.0
     
        return images,labels

    def load_mnist_test(self):
        file1 = 't10k-labels.idx1-ubyte'
        file2 = 't10k-images.idx3-ubyte'
        labels = idx2numpy.convert_from_file(file1)
        images = idx2numpy.convert_from_file(file2)
        images = images/255.0
     
        return images,labels


    def print_data(self):
        print(self.input_data)

start = time.time()

weights = np.matrix('0.1 0.1 -0.3;0.1 0.2 0.0;0.0 0.7 0.1;0.2 0.4 0.0')
alpha = 0.005
weights_range = [-0.1, 0.1]
# matrix_size = [3, 4]
input_size = [784,1]
hidden_weights_size = [100, 784]
output_weights_size = [10, 100]
how_many_iterations = 350
how_many_images = 10000

error_sum = 0


print("Wczytywanie pliku")
myNetwork2 = neuralNetwork(hidden_weights_size, weights_range)
myNetwork2.add_layer_rand(output_weights_size, weights_range=weights_range)

print(myNetwork2.data[0].shape)
print(myNetwork2.data[1].shape)
print("Zakres wag: " + str(weights_range))

images, labels = myNetwork2.load_mnist()
expected = []
for i in range(60000):
    temp = np.asmatrix(np.zeros((10, 1)))
    temp[(labels[i])]=1
    expected.append(temp)

images = images.reshape((60000,784,1))
myNetwork2.input_data.clear()
myNetwork2.input_data.append(images)
# odpowiednik fit


myNetwork2.save_weights_og()
print("Treningowe Dane:")
correct, res=myNetwork2.update_weights(expected,alpha,cycles=how_many_iterations,how_many_img=how_many_images,percent_to_erase=50)
myNetwork2.save_weights()
print("Testowe Dane:")

images, labels = myNetwork2.load_mnist_test()
expected.clear()
# przygotowanie listy z expected w 0/1
for i in range(10000):
    temp = np.asmatrix(np.zeros((10, 1)))
    temp[(labels[i])]=1
    expected.append(temp)

images = images.reshape((10000,784,1))
myNetwork2.input_data.clear()
myNetwork2.input_data.append(images)
myNetwork2.smart_neural_network(expected=expected)

end = time.time()
print("Operation took:")
print(str(end-start) + (" seconds"))






