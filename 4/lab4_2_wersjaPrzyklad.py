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
        return (np.asmatrix(self.data[layer_number]) * np.asmatrix(input_vector))
        # return np.asmatrix(self.data[layer_number]) * np.asmatrix(input_vector)

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
                # pętla dla wag hidden -> output
                # tutaj jest problem
                batch = np.asmatrix(self.input_data[0][int(a*batch_size):int((a+1)*batch_size)])
                # batch = np.asmatrix(self.input_data[0][int(a*batch_size):int((a+1)*batch_size)]).T
# 784,100       
                # if np.array_equal(batch[:,3], self.input_data[0][(a*batch_size)+3]):
                #     print("tru")

                layer_hidden_1 = self.predict(np.asmatrix(batch),0)
                layer_hidden_1 = self.ReLU(layer_hidden_1)
                dropout_mask = np.random.randint(2,size=layer_hidden_1.shape)
                
                # layer_hidden_1 = np.multiply(layer_hidden_1,dropout_mask)*2


                layer_output = self.predict(layer_hidden_1,1)
                layer_output_delta = (2/layer_output.shape[0]*(layer_output-expected))/batch_size

                layer_hidden_1_delta = self.data[1].T * layer_output_delta
                deriv_relu_fun = np.vectorize(self.ReLU_deriv)
                layer_hidden_1_delta = np.multiply(layer_hidden_1_delta, deriv_relu_fun(layer_hidden_1))
                # layer_hidden_1_delta = np.multiply(layer_hidden_1_delta,dropout_mask)

                layer_output_weight_delta = np.asmatrix(layer_output_delta) * np.asmatrix(layer_hidden_1).T
                layer_hidden_1_weight_delta = np.asmatrix(layer_hidden_1_delta) * np.asmatrix(batch).T


                # to do poprawki!!!!!!!
                
                test = np.argmax(expected[a])
                # możliwe że da sie to zrobić z np.where
                for i in range(batch_size):

                    test = np.argmax(expected[a][i].T)

                    maxRow = layer_output[0, i]
                    maxIndex = 0
                    for j in range(10):
                        if layer_output[j, i] >= maxRow:
                            maxRow = layer_output[j, i]
                            maxIndex = j
                    if expected[a][maxIndex,0] == 1:
                        correct_counter += 1
                    how_many_possible+=1
                self.data[0] = np.asmatrix(self.data[0]) - np.asmatrix(alpha * layer_hidden_1_weight_delta)
                self.data[1] = np.asmatrix(self.data[1]) - np.asmatrix(alpha * layer_output_weight_delta)

            print("Procent poprawnych: " + "numer serii:  " + str(b))
            print(correct_counter/(how_many_img*batch_size)*100)
            print(correct_counter)
            print(how_many_possible)

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

    def loadTwoFiles(self,file1,file2):
        temp1 = np.asmatrix(np.loadtxt(file1))
        temp2 = np.asmatrix(np.loadtxt(file2))
        self.data.append(temp1)
        self.data.append(temp2)

    def print_data(self):
        print(self.input_data)

start = time.time()

alpha = 0.1
weights_range = [-0.1, 0.1]
# matrix_size = [3, 4]
input_size = [784,1]
hidden_weights_size = [5, 3]
output_weights_size = [3, 5]
how_many_iterations = 1
how_many_images = 1
batch_size = 4
error_sum = 0


print("Wczytywanie pliku")
myNetwork2 = neuralNetwork(hidden_weights_size, weights_range)
myNetwork2.data.clear()
myNetwork2.loadTwoFiles("weights_hidden.txt","weights.txt")
# myNetwork2.add_layer_rand(output_weights_size, weights_range=weights_range)

input = np.asmatrix(np.loadtxt("input.txt"))
myNetwork2.input_data.append(input)

print(myNetwork2.data[0].shape)
print(myNetwork2.data[1].shape)
# print("Zakres wag: " + str(weights_range))

expected = np.asmatrix(np.loadtxt("expected.txt"))

# odpowiednik fit

myNetwork2.save_weights_og()
print("Treningowe Dane:")
correct, res=myNetwork2.update_weights(expected,alpha,cycles=how_many_iterations,how_many_img=batch_size,batch_size=batch_size,percent_to_erase=50)
myNetwork2.save_weights()
print("Testowe Dane:")

# images, labels = myNetwork2.load_mnist_test()
# expected.clear()
# # przygotowanie listy z expected w 0/1
# for i in range(10000):
#     temp = np.asmatrix(np.zeros((10, 1)))
#     temp[(labels[i])]=1
#     expected.append(temp)

# images = images.reshape((10000,784,1))
myNetwork2.input_data.clear()
myNetwork2.input_data.append(input)
myNetwork2.smart_neural_network(expected=expected)

end = time.time()
print("Operation took:")
print(str(end-start) + (" seconds"))






# & C:/Users/Mati/AppData/Local/Programs/Python/Python310/python.exe d:/Politechnika/Python/PSI/lab4/lab4_2.py