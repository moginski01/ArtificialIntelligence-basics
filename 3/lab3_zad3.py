import numpy as np

class neuralNetwork:
    data = []
    input_data=[]
    error = 0

    def __init__(self, first_matrix_size, weights_range):
        temp = np.random.uniform(weights_range[0],weights_range[1], size=(first_matrix_size[0], first_matrix_size[1]))
        self.data.append(temp)

    def add_layer(self, weights):
        if len(self.data) == 0:
            self.data.append(weights)

    def ReLU_deriv(self,x):
        if x>0:
            return 1
        else:
            return 0

    def ReLU(self,x):
        return np.maximum(0, x)
    

    def add_layer_rand(self, weights_range):
        temp = np.asmatrix(np.random.uniform(weights_range[0],weights_range[1], size=(self.data[-1].shape[0],self.data[-1].shape[0])))
        self.data.append(temp)

    def clear_data(self):
        self.data.clear()

    def predict(self, input_vector, layer_number):
        debug1= input_vector
        debug2 = self.data[layer_number]
        # if layer_number==0:
        #     # tylko po to by w takim samym "formacie" były
        #     return (self.data[layer_number]*input_vector.T)
        # else:
        return self.data[layer_number]*input_vector
        

    def update_weights(self, expected, alpha, cycles):        
        
        correct_counter=0
        # print(len(self.data))
        out = []
        temp = 0
        for b in range(cycles):
            correct_counter=0
            # a numer serri
            # pętla dla serii
            for a in range(len(self.input_data)):
                out = []
                #pętla dla wag hidden -> output
                for i in range(len(self.data)):
                    if i == 0:

                        temp = self.predict(np.asmatrix(self.input_data[a][0:3]).T,0)
                        temp = self.ReLU(temp)
                        out.append(temp)
                    else:
                        temp = self.predict(np.asmatrix(out[-1]),1)
                        out.append(temp)
                                 
                        layer_output_delta = (2/out[-1].shape[0]) * (out[-1] - expected[a].T)
                        layer_hidden_1_delta = self.data[i].T * layer_output_delta
                        temp_vec_function = np.vectorize(self.ReLU_deriv)

                        # print(out)
                        layer_hidden_1_delta = np.multiply(layer_hidden_1_delta,temp_vec_function(out[-2]))
                        
                        layer_output_weight_delta =np.asmatrix(layer_output_delta) * np.asmatrix(out[-2]).T
                        layer_hidden_1_weight_delta = np.asmatrix(layer_hidden_1_delta) * np.asmatrix(self.input_data[a][0:3])

                        maxRow = out[-1][0,0]
                        maxIndex = 0
                        thisIterationCorrect = 0
                        # pętla do testu czy dobrze zgadło
                        for j in range(4):
                            if out[-1][j,0]>=maxRow:
                                maxRow=out[-1][j,0]
                                maxIndex = j
                        if expected[a][0,maxIndex]==1:
                            correct_counter+=1
                            thisIterationCorrect = 1
                        # else:
                        self.data[0] = np.asmatrix(self.data[0]) - np.asmatrix(alpha*layer_hidden_1_weight_delta)
                        self.data[1] = np.asmatrix(self.data[1]) - np.asmatrix(alpha*layer_output_weight_delta)       

        return correct_counter,cycles


    def smart_neural_network(self, expected):
        correct_counter=0
        # print(len(self.data))
        out = []
        temp = 0
    
        correct_counter=0
        # a numer serri
        # pętla dla serii
        for a in range(len(self.input_data)):
            out = []
            #pętla dla wag hidden -> output
            for i in range(len(self.data)):
                if i == 0:

                    temp = self.predict(np.asmatrix(self.input_data[a][0:3]).T,0)
                    temp = self.ReLU(temp)
                    out.append(temp)
                else:
                    temp = self.predict(np.asmatrix(out[-1]),1)
                    out.append(temp)
                                
                    maxRow = out[-1][0,0]
                    maxIndex = 0
                    # thisIterationCorrect = 0
                    # pętla do testu czy dobrze zgadło
                    for j in range(4):
                        if out[-1][j,0]>=maxRow:
                            maxRow=out[-1][j,0]
                            maxIndex = j
                    if expected[a][0,maxIndex]==1:
                        correct_counter+=1
                                        
        return correct_counter



    def load_weights(self, file_name):
        # np.savetxt('test1.txt', self.data[0])
        weights1 = np.loadtxt('wagi1og_zad3.txt')
        weights2 = np.loadtxt('wagi2og_zad3.txt')
        # weights1 = np.loadtxt('weights1.txt')
        # weights2 = np.loadtxt('weights2.txt')
        self.data.append(weights1)
        self.data.append(weights2)
        # print(temp)
    def save_weights_og(self):
        np.savetxt("wagi1og_zad3.txt",self.data[0])
        np.savetxt("wagi2og_zad3.txt",self.data[1])

    
    def load_multiple_weights(self, file_name):
        print("funload")
        temp = np.loadtxt(file_name)
        temp_matrix = np.zeros((4,3))
        temp_matrix = np.asmatrix(temp_matrix)
        how_many_matrixes = float(temp.shape[0]) / float(temp_matrix.shape[1])

        for i in range(int(how_many_matrixes)):
            temp_matrix = temp[(i*3):((i+1)*3),:]
            self.data.append(temp_matrix)
        

    def load_multiple_inputs(self,file_name):
        temp = np.loadtxt(file_name)
        temp_matrix = np.zeros((1,4))
        temp_matrix = np.asmatrix(temp_matrix)
        how_many_matrixes = float(temp.shape[0])
        # print(how_many_matrixes)
        for i in range(int(how_many_matrixes)):
            temp_matrix = temp[i,:]
            # print(temp_matrix)
            self.input_data.append(temp_matrix)

    def print_data(self):
        print(self.input_data)           


# weights = np.matrix('0.1 0.1 -0.3;0.1 0.2 0.0;0.0 0.7 0.1;0.2 0.4 0.0')
alpha = 0.01
weights_range = [-0.6, 0.6]
# matrix_size = [3, 4]
matrix_size = [4, 3]
error_sum = 0

print("Wczytywanie pliku")
myNetwork2 = neuralNetwork(matrix_size, weights_range)
myNetwork2.add_layer_rand(weights_range)
# myNetwork2.save_weights_og()#zapisane są okej wyniki 
myNetwork2.load_multiple_inputs("training_colors.txt")

myNetwork2.data.clear()
myNetwork2.load_weights("placeholder.txt")

expected_weights=[]
# expected_weights = np.asmatrix(expected_weights)
one = np.zeros((1,4))
two = np.zeros((1,4))
three = np.zeros((1,4))
four = np.zeros((1,4))
one=np.asmatrix(one)
two = np.asmatrix(two)
three = np.asmatrix(three)
four = np.asmatrix(four)
one[0, 0] = 1
two[0, 1] = 1
three[0, 2] = 1
four[0, 3] = 1

for i in range(len(myNetwork2.input_data)):
    # for j in range(4):
        # temp=np.zeros((1,4))
        # print(int(myNetwork2.input_data[j][-1]-1))
        # temp[0,int(myNetwork2.input_data[j][-1]-1)]=1
        # expected_weights.append(np.asmatrix(temp))
  
   
    if myNetwork2.input_data[i][-1]==1:
        expected_weights.append(one)
    elif myNetwork2.input_data[i][-1]==2:
        expected_weights.append(two)
    elif myNetwork2.input_data[i][-1]==3:
        expected_weights.append(three)
    elif myNetwork2.input_data[i][-1]==4:
        expected_weights.append(four)


# print(myNetwork2.data[0])
# print(myNetwork2.data[1])

correct,cycles=myNetwork2.update_weights(expected_weights,alpha,20)
print(correct)
print("Ilość cykli:")
print(cycles)
print("Procent poprawnych")
print(correct/109*100)

#teraz dane testowe
print("DANE TESTOWE:")
myNetwork2.input_data.clear()
expected_weights.clear()

myNetwork2.load_multiple_inputs("test_colors.txt")
# print(len(myNetwork2.input_data))
for i in range(len(myNetwork2.input_data)):
    # print(myNetwork2.input_data[i])
    # print(int(myNetwork2.input_data[i][-1]))
    if myNetwork2.input_data[i][-1]==1:
        expected_weights.append(one)
    elif myNetwork2.input_data[i][-1]==2:
        expected_weights.append(two)
    elif myNetwork2.input_data[i][-1]==3:
        expected_weights.append(three)
    elif myNetwork2.input_data[i][-1]==4:
        expected_weights.append(four)

correct = myNetwork2.smart_neural_network(expected_weights)
# print(correct)
print("Procent poprawnych")
print(correct/130*100)


