from email import iterators
import numpy as np

def ReLU(x):
    return np.maximum(0, x)
    
def ReLU_deriv(x):
    if x>0:
        return 1
    else:
        return 0
    


def deep_neural_network(input_vector, weights, expected, alpha, iterations):
    temp_inp = np.zeros((5,1))
    temp_inp = np.asmatrix(temp_inp)
    # print(input_vector.shape[0])
    out = []
    # for a in range(input_vector.shape[0]):
    for b in range(iterations):
        out.clear()#po to by tylko ostatnia epoka została
        for a in range(4):
        # 4 bo tyle serii danych

            for i in range(len(weights)):

                if i==0:
                    temp = weights[i] * input_vector[:,a]
                    # print(temp)
                    temp = ReLU(temp)
                    temp_inp = temp
                    out.append(temp)
                else:
                    temp = weights[i] * out[-1]
                    
                    # print(temp)
                    temp_shape = temp.shape
                    # input_vector[:,a] = temp
                    out.append(temp)

                    # aktualizacja wag dla ukrytej i output layer

                    layer_output_delta = (2/out[-1].shape[0]) * (out[-1] - expected[:,a])
                    layer_hidden_1_delta = weights[i].T * layer_output_delta
                    temp_vec_function = np.vectorize(ReLU_deriv)

                    # print(out)
                    layer_hidden_1_delta = np.multiply(layer_hidden_1_delta,temp_vec_function(out[-2]))
                    
                    layer_output_weight_delta =layer_output_delta * out[-2].T
                    layer_hidden_1_weight_delta = layer_hidden_1_delta * input_vector[:,a].T
                    # test_inputa = input_vector[:,a].T

                    weights[0] = weights[0] - alpha*layer_hidden_1_weight_delta
                    weights[1] = weights[1] - alpha*layer_output_weight_delta

                    # print("STOP")

    return out



weights_hidden1 = np.matrix('0.1 0.1 -0.3;0.1 0.2 0.0;0.0 0.7 0.1;0.2 0.4 0.0;-0.3 0.5 0.1')
weights1 = np.matrix('0.7 0.9 -0.4 0.8 0.1;0.8 0.5 0.3 0.1 0.0;-0.3 0.9 0.3 0.1 -0.2')

inp = np.matrix('0.5 0.1 0.2 0.8;0.75 0.3 0.1 0.9;0.1 0.7 0.6 0.2')
expected1 = np.matrix('0.1 0.5 0.1 0.7;1.0 0.2 0.3 0.6;0.1 -0.5 0.2 0.2')

alpha = 0.01
scales = []
scales.append(weights_hidden1)
scales.append(weights1)
iterations = 1
out = deep_neural_network(inp, scales, expected1, alpha, iterations)
# print(len(out))
print("Po 1 epoce")
for i in range(len(out)):
    if i % 2 == 1:
        print(out[i])

scales.clear()
scales.append(weights_hidden1)
scales.append(weights1)
iterations = 50
out = deep_neural_network(inp, scales, expected1, alpha, iterations)
print()
print("Po 50 epokach")
for i in range(len(out)):
    if i % 2 ==1:
        print(out[i])