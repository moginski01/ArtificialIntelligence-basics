import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import time
import random
import math

class neuralNetwork:
    convol_weights = []
    out_weights = []
    input_data = []
    error = 0
    labels_dec = []
    kernels2 = []
    test_input_data = []
    test_expected = []


    def __init__(self, first_matrix_size, weights_range):
      
        temp = np.random.uniform(weights_range[0], weights_range[1], size=(first_matrix_size[0], first_matrix_size[1]))
        self.convol_weights.append(temp)

    def add_layer(self, weights):
        if len(self.convol_weights) == 0:
            self.convol_weights.append(weights)

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
        self.convol_weights.append(temp)

    def add_layer_out_rand(self, size, weights_range):
        temp = np.asmatrix(np.random.uniform(weights_range[0], weights_range[1], size=(size[0], size[1])))
        # new_shape = temp.shape
        self.out_weights.append(temp)

    def clear_data(self):
        self.convol_weights.clear()

    def predict(self, input_vector, layer_number):
        return np.asmatrix(self.convol_weights[layer_number]) * np.asmatrix(input_vector)

    def fit_convol(self,expected,alpha,iterations,images_count):
        # kernel = np.copy(self.convol_weights[0])
        kernel = np.asmatrix(np.zeros((1,9)))
        for i in range(3):
            kernel[0,(i*3):((i+1)*3)] = self.convol_weights[0][:,i].T
        # kernels.append(kernel2)
        size = (int(self.input_data[0].shape[1]- int(math.sqrt(kernel.shape[1])))+1)
        kernels = []
        # kernels.append(kernel)
        for j in range(len(self.convol_weights)):
            # for i in range(3):
            #     kernel[0,(i*3):((i+1)*3)] = self.convol_weights[j][:,i].T
            kernel = self.convol_weights[j].T.reshape(-1,9)
            kernels.append(kernel)
            # kernels.append(self.convol_weights[i])

        kernels = np.asmatrix(np.asarray(kernels))
        how_many_sections = size**2

        kernels_2 = []
        for i in range(16):
            kernels_2.append(self.convol_weights[i])

        kernels_2 = np.asarray(kernels_2)
    

        # for a in range(self.input_data[0].shape[0]):
        for b in range(iterations):
            correct_counter=0
            for a in range(images_count):
                section_row = 0
                # image_sections = np.zeros((size,size,16))
                image_sections = np.asmatrix(np.zeros((how_many_sections,self.convol_weights[0].shape[1]**2)))
                image_sections2 = np.zeros((676,3,3))
                kernel_layer_2_shape = [size,size,16]
                kernel_layer_2 = np.zeros((kernel_layer_2_shape),dtype=float)
                for i in range(size):
                    for j in range(size):
                        # image_sections[section_row,:] = self.input_data[0][a,i:(i+3),j:(j+3)].T.reshape(-1,9)
                        image_sections[section_row,:] = self.input_data[0][a,i:(i+3),j:(j+3)].T.reshape(-1,9)
                        image_sections2[section_row,:] = self.input_data[0][a,i:(i+3),j:(j+3)]
                        #operacja konwolucji
                        for k in range(16):
                            kernel_layer_2[i,j,k] = np.sum(np.multiply(image_sections2[section_row,:],kernels_2[k,:,:]))
                    
                        section_row+=1

                
                kernel_layer_2 = self.ReLU(kernel_layer_2)

              
                #czyli tutaj chcemy 28x28x16
                # kernel_layer_reshape = np.asarray((np.dot(image_sections,kernels.T)))
                # kernel_layer_reshape = self.ReLU(kernel_layer_reshape)

                # test1 = np.arange((676))

                # kernel_layer = np.asarray((np.dot(image_sections,kernels.T)))
                # kernel_layer = kernel_layer.reshape(size,size,-1)
                # kernel_layer = self.ReLU(kernel_layer)
                #obstawiam tutaj wstawienie poolingu



                #do tąd przetestowane

                pool_layer_shape_0 = int(int(kernel_layer_2.shape[0])/2)
                pool_layer_shape_1 = int(int(kernel_layer_2.shape[1])/2)
                pool_layer_shape_2 = int(int(kernel_layer_2.shape[2]))
                pool_layer_shape = [pool_layer_shape_0,pool_layer_shape_1,pool_layer_shape_2]
                pool_layer = (np.zeros(pool_layer_shape))
                remember_mask = (np.zeros(kernel_layer_2.shape))

                           
                for aa in range(kernel_layer_2.shape[2]):
                    for ii in range(pool_layer.shape[0]):
                        for jj in range(pool_layer.shape[1]):
                            max_index = np.argmax(kernel_layer_2[ii*2:ii*2+2,jj*2:jj*2+2,aa])
                            pool_layer[ii,jj] = np.max(kernel_layer_2[ii*2:ii*2+2,jj*2:jj*2+2,aa])
                            x1 = max_index%2
                            y1 = int(max_index/2)
                            remember_mask[ii*2+y1,jj*2+x1,aa] = 1
                            # remember_mask[ii*2,jj*2] = 1


                # kernel_layer_flatten = kernel_layer.reshape(-1,1)
                kernel_layer_flatten = pool_layer.reshape(-1,1)
                # pool_layer_flatten = pool_layer.reshape(-1,1)

                #tu dropout
                # dropout_mask = np.random.randint(2,size=kernel_layer_flatten.shape)
                # kernel_layer_flatten = np.multiply(kernel_layer_flatten,dropout_mask)

                layer_output = self.out_weights[0].T * kernel_layer_flatten
                # layer_output = self.out_weights[0] * kernel_layer


                #Na razie do tąd!!!!
                N = 10
                layer_output_delta = (2*1/N *(np.matrix(layer_output)-np.matrix(expected[a])))
                #dla ukrytych do przemyślenia bo jest ich 16 teraz
                kernel_layer_1_delta = np.matrix(self.out_weights[0]) * np.matrix(layer_output_delta) 
                
                deriv_relu_fun = np.vectorize(self.ReLU_deriv)
                kernel_layer_1_delta = np.multiply(kernel_layer_1_delta,deriv_relu_fun(kernel_layer_flatten))
                # kernel_layer_1_delta = np.multiply(kernel_layer_1_delta,dropout_mask)

                #tu dropout
                kernel_layer_1_delta_reshaped = np.copy(kernel_layer_1_delta).reshape(pool_layer.shape)
                # kernel_layer_1_delta_reshaped = np.copy(kernel_layer_1_delta).reshape(kernel_layer.shape)

                #tutaj zrobić rozmiar z powrotem
                kernel_layer_1_delta_reshaped_og = np.zeros(kernel_layer_2.shape)
                for ii in range(pool_layer.shape[0]):
                    for jj in range(pool_layer.shape[1]):
                        for aa in range(pool_layer.shape[2]):
                            kernel_layer_1_delta_reshaped_og[ii*2:ii*2+2,jj*2:jj*2+2,aa] = kernel_layer_1_delta_reshaped[ii,jj,aa]

                kernel_layer_1_delta_reshaped_og = np.multiply(kernel_layer_1_delta_reshaped_og,remember_mask)

                layer_output_weight_delta = layer_output_delta * kernel_layer_flatten.T

                # kernel_layer_1_weight_delta = kernel_layer_1_delta_reshaped.T * image_sections
                #tutaj testowe flatten
                kernel_layer_1_delta_reshaped_og_flatten = kernel_layer_1_delta_reshaped_og.reshape(-1,16)
                kernel_layer_1_weight_delta = kernel_layer_1_delta_reshaped_og_flatten.T * image_sections


                #miejsce na sprawdzanie poprawnych danych
                if np.argmax(layer_output) == np.argmax(expected[a]):
                    correct_counter+=1
                self.out_weights[0] = np.asmatrix(self.out_weights[0]) - alpha*np.asmatrix(layer_output_weight_delta).T
                kernels = np.asmatrix(kernels) - alpha*np.asmatrix(kernel_layer_1_weight_delta)
                kernels_2 = np.asarray(kernels).reshape(kernels_2.shape)
                
                # print(a)

            print('procent poprawnych iteracja ' + str(b) + ' : ')
            print((correct_counter/images_count)*100)

        for i in range(16):
            self.kernels2.append(kernels_2[i])

        # print(kernel_layer)
        return 1

    def convol_test(self,expected):
                # kernel = np.copy(self.convol_weights[0])
        kernel = np.asmatrix(np.zeros((1,9)))
        for i in range(3):
            kernel[0,(i*3):((i+1)*3)] = self.convol_weights[0][:,i].T
        # kernels.append(kernel2)
        size = (int(self.input_data[0].shape[1]- int(math.sqrt(kernel.shape[1])))+1)
        kernels = []
        # kernels.append(kernel)
        for j in range(len(self.convol_weights)):
            # for i in range(3):
            #     kernel[0,(i*3):((i+1)*3)] = self.convol_weights[j][:,i].T
            kernel = self.convol_weights[j].T.reshape(-1,9)
            kernels.append(kernel)
            # kernels.append(self.convol_weights[i])

        kernels = np.asmatrix(np.asarray(kernels))
        how_many_sections = size**2

        kernels_2 = []
        for i in range(16):
            kernels_2.append(self.kernels2[i])

        kernels_2 = np.asarray(kernels_2)
    

        # for a in range(self.input_data[0].shape[0]):
        for b in range(1):
            correct_counter=0
            for a in range(10000):
                section_row = 0
                # image_sections = np.zeros((size,size,16))
                image_sections = np.asmatrix(np.zeros((how_many_sections,self.convol_weights[0].shape[1]**2)))
                image_sections2 = np.zeros((676,3,3))
                kernel_layer_2_shape = [size,size,16]
                kernel_layer_2 = np.zeros((kernel_layer_2_shape),dtype=float)
                for i in range(size):
                    for j in range(size):
                        # image_sections[section_row,:] = self.input_data[0][a,i:(i+3),j:(j+3)].T.reshape(-1,9)
                        image_sections[section_row,:] = self.input_data[0][a,i:(i+3),j:(j+3)].T.reshape(-1,9)
                        image_sections2[section_row,:] = self.input_data[0][a,i:(i+3),j:(j+3)]
                        #operacja konwolucji
                        for k in range(16):
                            kernel_layer_2[i,j,k] = np.sum(np.multiply(image_sections2[section_row,:],kernels_2[k,:,:]))
                    
                        section_row+=1

                
                kernel_layer_2 = self.ReLU(kernel_layer_2)

              
                #czyli tutaj chcemy 28x28x16
                # kernel_layer_reshape = np.asarray((np.dot(image_sections,kernels.T)))
                # kernel_layer_reshape = self.ReLU(kernel_layer_reshape)

                # test1 = np.arange((676))

                # kernel_layer = np.asarray((np.dot(image_sections,kernels.T)))
                # kernel_layer = kernel_layer.reshape(size,size,-1)
                # kernel_layer = self.ReLU(kernel_layer)
                #obstawiam tutaj wstawienie poolingu



                #do tąd przetestowane

                pool_layer_shape_0 = int(int(kernel_layer_2.shape[0])/2)
                pool_layer_shape_1 = int(int(kernel_layer_2.shape[1])/2)
                pool_layer_shape_2 = int(int(kernel_layer_2.shape[2]))
                pool_layer_shape = [pool_layer_shape_0,pool_layer_shape_1,pool_layer_shape_2]
                pool_layer = (np.zeros(pool_layer_shape))
                remember_mask = (np.zeros(kernel_layer_2.shape))

                           
                for aa in range(kernel_layer_2.shape[2]):
                    for ii in range(pool_layer.shape[0]):
                        for jj in range(pool_layer.shape[1]):
                            max_index = np.argmax(kernel_layer_2[ii*2:ii*2+2,jj*2:jj*2+2,aa])
                            pool_layer[ii,jj] = np.max(kernel_layer_2[ii*2:ii*2+2,jj*2:jj*2+2,aa])
                            x1 = max_index%2
                            y1 = int(max_index/2)
                            remember_mask[ii*2+y1,jj*2+x1,aa] = 1
                            # remember_mask[ii*2,jj*2] = 1


                # kernel_layer_flatten = kernel_layer.reshape(-1,1)
                kernel_layer_flatten = pool_layer.reshape(-1,1)
                # pool_layer_flatten = pool_layer.reshape(-1,1)

                #tu dropout
                layer_output = self.out_weights[0].T * kernel_layer_flatten
                # layer_output = self.out_weights[0] * kernel_layer

                #miejsce na sprawdzanie poprawnych danych
                if np.argmax(layer_output) == np.argmax(expected[a]):
                    correct_counter+=1
               

            print('procent poprawnych iteracja ' + str(b) + ' : ')
            print((correct_counter/10000)*100)

      
        # print(kernel_layer)
        return 1
    

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

                layer_hidden_1_delta = self.convol_weights[1].T * layer_output_delta
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
                self.convol_weights[0] = np.asmatrix(self.convol_weights[0]) - np.asmatrix(alpha * layer_hidden_1_weight_delta)
                self.convol_weights[1] = np.asmatrix(self.convol_weights[1]) - np.asmatrix(alpha * layer_output_weight_delta)

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

                layer_hidden_1_delta = self.convol_weights[1].T * layer_output_delta
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
            print(correct_counter/(how_many_img*batch_size))
            print(correct_counter)
            print(how_many_possible)

        return correct_counter, res


    def convol(self, step, padding):

        kernel = np.copy(self.convol_weights[0])
        kernel = np.asmatrix(np.zeros((1,9)))
        for i in range(3):
            kernel[0,(i*3):((i+1)*3)] = self.convol_weights[0][:,i].T


        # kernels.append(kernel2)
        size = (int(self.input_data[0].shape[0]- int(math.sqrt(kernel.shape[1])))+1)
        kernels = []
        # kernels.append(kernel)
        for i in range(size**2):
            kernels.append(kernel)

        kernels = np.asmatrix(np.asarray(kernels))
        how_many_sections = size**2

        #rozmiar img_sections będzie prawodpodobnie na podstawie expected size
        image_sections = np.asmatrix(np.zeros((how_many_sections,self.convol_weights[0].shape[1]**2)))

        # image sections
        section_row = 0
        for i in range(size):
            temp_list = []
            for j in range(size):
                temp = self.input_data[0][i:(i+3),j]
                image_sections[section_row,:] = self.input_data[0][i:(i+3),j:(j+3)].T.reshape(-1,9)
                section_row+=1

        kernel_layer = np.dot(image_sections,kernel.T).reshape(self.convol_weights[0].shape)

        return kernel_layer



    def save_weights_og(self):
        np.savetxt("wagi1og.txt",self.convol_weights[0])
        np.savetxt("wagi2og.txt",self.convol_weights[1])

    def save_weights(self):
        np.savetxt("wagi1.txt",self.convol_weights[0])
        np.savetxt("wagi2.txt",self.convol_weights[1])

    def load_weights(self, file_name):
        # np.savetxt('test1.txt', self.data[0])
        weights1 = np.asmatrix(np.loadtxt(file_name))
        # og_shape = weights1.shape
        # weights1 = weights1.reshape(1,og_shape[0]*og_shape[1])
        self.convol_weights.append(weights1)
        self.convol_weights.append(weights1)
        # print(temp)

    def load_multiple_weights(self, file_name):
        print("funload")
        temp = np.loadtxt(file_name)
        temp_matrix = np.zeros((4, 3))
        temp_matrix = np.asmatrix(temp_matrix)
        how_many_matrixes = float(temp.shape[0]) / float(temp_matrix.shape[1])

        for i in range(int(how_many_matrixes)):
            temp_matrix = temp[(i * 3):((i + 1) * 3), :]
            self.convol_weights.append(temp_matrix)

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



mask_weights_range = [-0.01,0.01]
out_weights_range = [-0.1,0.1]
mask_shape = [3,3]
# output_weights_shape = [10816,10]# cos zamiast tej "1"
output_weights_shape = [2704,10]# cos zamiast tej "1"
myNetwork = neuralNetwork(mask_shape,mask_weights_range)
myNetwork.convol_weights.clear()


images,labels = myNetwork.load_mnist()
expected = []
for i in range(60000):
    temp = np.asmatrix(np.zeros((10, 1)))
    temp[(labels[i])]=1
    expected.append(temp)

myNetwork.input_data.append(images)

#mialo byc 16 filtrow w warstwie konwol
for i in range(16):
    myNetwork.add_layer_rand(size=mask_shape,weights_range=mask_weights_range)

myNetwork.add_layer_out_rand(size=output_weights_shape,weights_range=out_weights_range)

res = myNetwork.fit_convol(expected,alpha=0.01,iterations=50,images_count=60000)
# res = myNetwork.convol(0,1)
# myNetwork.load
# print(res)

images2, labels2= myNetwork.load_mnist_test()
expected.clear()
# przygotowanie listy z expected w 0/1
for i in range(10000):
    temp = np.asmatrix(np.zeros((10, 1)))
    temp[(labels2[i])]=1
    expected.append(temp)

myNetwork.input_data.clear()
myNetwork.input_data.append(images2)

res = myNetwork.convol_test(expected)
# print(res)


end = time.time()
print("Operation took:")
print(str(end-start) + (" seconds"))





#& C:/Users/Mati/AppData/Local/Programs/Python/Python310/python.exe d:/Politechnika/Python/PSI/Lab5/lab5_3.py
# & C:/Users/Mati/AppData/Local/Programs/Python/Python310/python.exe d:/Politechnika/Python/PSI/lab4/lab4_2.py