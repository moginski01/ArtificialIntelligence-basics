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

    def tanh(self,x):
        return np.tanh(x)

    def tanh_deriv(self, output):
        return 1 - (output ** 2)

    def softmax(self, x):
        temp = np.exp(x)
        return temp / np.sum(temp)
        # return temp / np.sum(temp, axis=1, keepdims=True)


    def add_layer_rand(self, size, weights_range):
        temp = np.asmatrix(np.random.uniform(weights_range[0], weights_range[1], size=(size[0], size[1])))
        # new_shape = temp.shape
        self.data.append(temp)

    def clear_data(self):
        self.data.clear()

    def predict(self, input_vector, layer_number):
        return np.asmatrix(self.data[layer_number]) * np.asmatrix(input_vector)

    def update_weights(self, expected, alpha, cycles, how_many_img, batch_size, percent_to_erase):

        correct_counter = 0
        # print(len(self.data))
        out = []
        res = []
        temp = 0
        # temp_shape = self.input_data[0][0].shape
        # na razie na 1 ustawione
        for b in range(cycles):
            correct_counter = 0
            how_many_possible = 0
            # a numer serri
            # pętla dla ilości batchy
            for a in range(int(how_many_img/batch_size)):
             
                batch = np.asmatrix(self.input_data[0][int(a*batch_size):int((a+1)*batch_size)]).T

                layer_hidden_1 = self.predict(np.asmatrix(batch),0)
                layer_hidden_1 = self.tanh(layer_hidden_1)
                # layer_hidden_1 = self.ReLU(layer_hidden_1)
                dropout_mask = np.random.randint(2,size=layer_hidden_1.shape)
                layer_hidden_1 = np.multiply(layer_hidden_1,dropout_mask)*2


                layer_output = self.predict(layer_hidden_1,1)
                #tutaj softmax
                for i in range(batch_size):
                    layer_output[:,i] = self.softmax(layer_output[:,i]);
                layer_output_delta = (2/layer_hidden_1.shape[0]*(layer_output-expected[a].T))/batch_size

                layer_hidden_1_delta = self.data[1].T * layer_output_delta
                deriv_relu_fun = np.vectorize(self.tanh_deriv)
                layer_hidden_1_delta = np.multiply(layer_hidden_1_delta, deriv_relu_fun(layer_hidden_1))
                layer_hidden_1_delta = np.multiply(layer_hidden_1_delta,dropout_mask)

                layer_output_weight_delta = np.asmatrix(layer_output_delta) * np.asmatrix(layer_hidden_1).T
                layer_hidden_1_weight_delta = np.asmatrix(layer_hidden_1_delta) * np.asmatrix(batch).T
                
                test = np.argmax(expected[a])
                for i in range(batch_size):
                    maxVal = np.argmax(layer_output[:,i])
                    expectedMax = np.argmax(expected[a][i,:].T)
                    if expectedMax == maxVal:
                        correct_counter+=1;

                    how_many_possible+=1
                self.data[0] = np.asmatrix(self.data[0]) - np.asmatrix(alpha * layer_hidden_1_weight_delta)
                self.data[1] = np.asmatrix(self.data[1]) - np.asmatrix(alpha * layer_output_weight_delta)

            print("Procent poprawnych: " + "numer serii:  " + str(b))
            print(correct_counter/(how_many_img*batch_size)*100)
            print(correct_counter)
            print(how_many_possible)

        return correct_counter, res

    def smart_neural_network(self, expected, cycles, how_many_img, batch_size):

        correct_counter = 0
        # print(len(self.data))
        out = []
        res = []
        temp = 0
        # temp_shape = self.input_data[0][0].shape
        # na razie na 1 ustawione
        for b in range(cycles):
            correct_counter = 0
            how_many_possible = 0
            # a numer serri
            # pętla dla ilości batchy
            for a in range(int(how_many_img/batch_size)):
             
                batch = np.asmatrix(self.input_data[0][int(a*batch_size):int((a+1)*batch_size)]).T

                layer_hidden_1 = self.predict(np.asmatrix(batch),0)
                layer_hidden_1 = self.ReLU(layer_hidden_1)
                # dropout_mask = np.random.randint(2,size=layer_hidden_1.shape)
                # layer_hidden_1 = np.multiply(layer_hidden_1,dropout_mask)*2


                layer_output = self.predict(layer_hidden_1,1)
                layer_output_delta = (2/layer_hidden_1.shape[0]*(layer_output-expected[a].T))/batch_size

                layer_hidden_1_delta = self.data[1].T * layer_output_delta
                deriv_relu_fun = np.vectorize(self.ReLU_deriv)
                layer_hidden_1_delta = np.multiply(layer_hidden_1_delta, deriv_relu_fun(layer_hidden_1))
                # layer_hidden_1_delta = np.multiply(layer_hidden_1_delta,dropout_mask)

                layer_output_weight_delta = np.asmatrix(layer_output_delta) * np.asmatrix(layer_hidden_1).T
                layer_hidden_1_weight_delta = np.asmatrix(layer_hidden_1_delta) * np.asmatrix(batch).T
                
                test = np.argmax(expected[a])
                for i in range(batch_size):
                    maxVal = np.argmax(layer_output[:,i])
                    expectedMax = np.argmax(expected[a][i,:].T)
                    if expectedMax == maxVal:
                        correct_counter+=1;

                    how_many_possible+=1
                # self.data[0] = np.asmatrix(self.data[0]) - np.asmatrix(alpha * layer_hidden_1_weight_delta)
                # self.data[1] = np.asmatrix(self.data[1]) - np.asmatrix(alpha * layer_output_weight_delta)

            print("Procent poprawnych: " + "numer serii:  " + str(b))
            print(correct_counter/(how_many_img*batch_size)*100)
            print(correct_counter)
            print(how_many_possible)

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
alpha = 0.1
weights_range = [-0.1, 0.1]
# matrix_size = [3, 4]
input_size = [784,1]
hidden_weights_size = [100, 784]
output_weights_size = [10, 100]
how_many_iterations = 350
how_many_images = 10000
batch_size = 100
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

expected_batch = []
for i in range(int(len(expected)/batch_size)):
    expected_batch.append(np.asmatrix(np.asarray((expected[(i*batch_size):((i+1)*batch_size)]))))


# print(expected_batch[1])
# print(expected_batch[1].shape)
# print(expected_batch[1][1])
# print(expected_batch[1][2])
# print(expected_batch[1][3])

images = images.reshape((60000,784,1))
myNetwork2.input_data.clear()
myNetwork2.input_data.append(images)
# odpowiednik fit

myNetwork2.save_weights_og()
print("Treningowe Dane:")
correct, res=myNetwork2.update_weights(expected_batch,alpha,cycles=how_many_iterations,how_many_img=how_many_images,batch_size=batch_size,percent_to_erase=50)
myNetwork2.save_weights()
print("Testowe Dane:")

images, labels = myNetwork2.load_mnist_test()
expected.clear()
# przygotowanie listy z expected w 0/1
for i in range(10000):
    temp = np.asmatrix(np.zeros((10, 1)))
    temp[(labels[i])]=1
    expected.append(temp)

expected_batch.clear()
# expected_batch = []
for i in range(int(len(expected)/batch_size)):
    expected_batch.append(np.asmatrix(np.asarray((expected[(i*batch_size):((i+1)*batch_size)]))))

images = images.reshape((10000,784,1))
myNetwork2.input_data.clear()
myNetwork2.input_data.append(images)
myNetwork2.smart_neural_network(expected_batch,cycles=1,how_many_img=10000,batch_size=batch_size)


end = time.time()
print("Operation took:")
print(str(end-start) + (" seconds"))






# & C:/Users/Mati/AppData/Local/Programs/Python/Python310/python.exe d:/Politechnika/Python/PSI/lab4/lab4_2.py