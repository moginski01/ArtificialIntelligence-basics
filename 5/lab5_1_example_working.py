import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import time
import random
import math

class neuralNetwork:
    weights = []
    input_data = []
    error = 0
    labels_dec = []

    def __init__(self, first_matrix_size, weights_range):
      
        temp = np.random.uniform(weights_range[0], weights_range[1], size=(first_matrix_size[0], first_matrix_size[1]))
        self.weights.append(temp)

    def add_layer(self, weights):
        if len(self.weights) == 0:
            self.weights.append(weights)

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
        self.weights.append(temp)

    def clear_data(self):
        self.weights.clear()

    def predict(self, input_vector, layer_number):
        return np.asmatrix(self.weights[layer_number]) * np.asmatrix(input_vector)

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
                # dropout_mask = np.random.randint(2,size=layer_hidden_1.shape)
                dropout = [0] * int(layer_hidden_1.shape[0]*layer_hidden_1.shape[1]*0.5) + [1] * int(layer_hidden_1.shape[0]*layer_hidden_1.shape[1]*0.5)
                random.shuffle(dropout)
                # print(dropout)
                dropout_mask = np.asmatrix(np.asarray(dropout)).reshape(layer_hidden_1.shape)

                layer_hidden_1 = np.multiply(layer_hidden_1,dropout_mask)*2


                layer_output = self.predict(layer_hidden_1,1)
                #tutaj softmax
                for i in range(batch_size):
                    layer_output[:,i] = self.softmax(layer_output[:,i]);
                layer_output_delta = (2/layer_hidden_1.shape[0]*(layer_output-expected[a].T))/batch_size

                layer_hidden_1_delta = self.weights[1].T * layer_output_delta
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
                self.weights[0] = np.asmatrix(self.weights[0]) - np.asmatrix(alpha * layer_hidden_1_weight_delta)
                self.weights[1] = np.asmatrix(self.weights[1]) - np.asmatrix(alpha * layer_output_weight_delta)

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

                layer_hidden_1_delta = self.weights[1].T * layer_output_delta
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


    def convol(self, step, padding):
        # size = (int(self.input_data[0].shape[0]- int(self.weights[0].shape[1]))+1)
        # kernel = self.weights[0].reshape(1,size**2)
        kernel = np.matrix('0.1 0.2 -0.1 -0.1 0.1 0.9 0.1 0.4 0.1')
        kernel2 = np.matrix('0.3 1.1 -0.3 0.1 0.2 0.0 0.0 1.3 0.1')
        kernels = []
        kernels.append(kernel)
        kernels.append(kernel2)
        size = (int(self.input_data[0].shape[0]- int(math.sqrt(kernel.shape[1])))+1)

        # for i in range(size**2):
        #     kernels.append(kernel)

        kernels = np.asmatrix(np.asarray(kernels))
        # how_many_sections = size**2
        how_many_sections = 2
        # res_img = np.asmatrix(np.zeros((weights_shape[1]*))
        #rozmiar img_sections będzie prawodpodobnie na podstawie expected size
        image_sections = np.asmatrix(np.zeros((how_many_sections,self.weights[0].shape[1]**2)))

        # image sections
        for i in range(size):
            temp_list = []
            for j in range(3):
                temp = self.input_data[0][i:(i+3),j]
                image_sections[i,(j*3):((j+1)*3)] = self.input_data[0][i:(i+3),j].T

        # for i in range(size):
        #     for j in range(size):
        #         res = self.input_data[0][i:(i+3+1),j:(j+3+1)]
        #         res_list.append(res)



        # res = np.asmatrix(np.asarray(res_list))
        kernel_layer = image_sections * kernels.T
        kernel_layer_dot = np.dot(image_sections,kernels.T)
        # conv_res = res * kernels.T

        # res = np.multiply(self.input_data[0][0:3,0:3],self.weights[0])
        print("placeholder Convol")



    def save_weights_og(self):
        np.savetxt("wagi1og.txt",self.weights[0])
        np.savetxt("wagi2og.txt",self.weights[1])

    def save_weights(self):
        np.savetxt("wagi1.txt",self.weights[0])
        np.savetxt("wagi2.txt",self.weights[1])

    def load_weights(self, file_name):
        # np.savetxt('test1.txt', self.data[0])
        weights1 = np.asmatrix(np.loadtxt(file_name))
        # og_shape = weights1.shape
        # weights1 = weights1.reshape(1,og_shape[0]*og_shape[1])
        self.weights.append(weights1)
        self.weights.append(weights1)
        # print(temp)

    def load_multiple_weights(self, file_name):
        print("funload")
        temp = np.loadtxt(file_name)
        temp_matrix = np.zeros((4, 3))
        temp_matrix = np.asmatrix(temp_matrix)
        how_many_matrixes = float(temp.shape[0]) / float(temp_matrix.shape[1])

        for i in range(int(how_many_matrixes)):
            temp_matrix = temp[(i * 3):((i + 1) * 3), :]
            self.weights.append(temp_matrix)

    def load_input(self, file_name):
        inp = np.asmatrix(np.loadtxt(file_name))
        self.input_data.append(inp)
        # print("placeholder")
        


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

weights_range = [-0.1,0.1]
weights_shape = [3,4]
myNetwork = neuralNetwork(weights_shape, weights_range)
myNetwork.weights.clear()

myNetwork.load_input("input_image_example.txt")
myNetwork.load_weights("filter.txt")
myNetwork.convol(0,1)
# myNetwork.load





end = time.time()
print("Operation took:")
print(str(end-start) + (" seconds"))






# & C:/Users/Mati/AppData/Local/Programs/Python/Python310/python.exe d:/Politechnika/Python/PSI/lab4/lab4_2.py