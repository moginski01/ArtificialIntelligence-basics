import numpy as np


# def deep_neural_network(input_vector, scales):
#     # print("test")
#     for i in range(len(scales)):
#         input_shape = input_vector.shape
#         matrix_shape = scales[i].shape

#         if matrix_shape[1] == input_shape[0] or matrix_shape[0] == input_shape[0]:
#             input_vector = scales[i] * input_vector
#         else:
#             print("Wrong size of scales or input_vector")
#             return -1;

#     return input_vector

def ReLU(x):
    return np.maximum(0, x)
    


def deep_neural_network(input_vector, weights):
    temp_inp = np.zeros((5,1))
    temp_inp = np.asmatrix(temp_inp)
    # print(input_vector.shape[0])
    out = []
    # for a in range(input_vector.shape[0]):
    for a in range(4):
        
        for i in range(len(weights)):

            if i==0:
                input_shape = input_vector[:,a].shape
                matrix_shape = weights[i].shape
                # print(weights[i])
                # print(input_vector[:,a])

                temp = weights[i] * input_vector[:,a]
                # print(temp)
                temp_shape = temp.shape
                # input_vector[:,a] = temp
                temp = ReLU(temp)
                temp_inp = temp
                # out.append(temp)
            else:
                input_shape = temp_inp.shape
                matrix_shape = weights[i].shape
                # print(weights[i])
                # print(input_vector[:,a])

                temp = weights[i] * temp_inp
                
                # print(temp)
                temp_shape = temp.shape
                # input_vector[:,a] = temp
                temp_inp = temp
                out.append(temp)
        # print(out[a])
    return out



weights_hidden1 = np.matrix('0.1 0.1 -0.3;0.1 0.2 0.0;0.0 0.7 0.1;0.2 0.4 0.0;-0.3 0.5 0.1')
weights1 = np.matrix('0.7 0.9 -0.4 0.8 0.1;0.8 0.5 0.3 0.1 0.0;-0.3 0.9 0.3 0.1 -0.2')

inp = np.matrix('0.5 0.1 0.2 0.8;0.75 0.3 0.1 0.9;0.1 0.7 0.6 0.2')

scales = []
scales.append(weights_hidden1)
scales.append(weights1)

# print(len(scales))
out = deep_neural_network(inp, scales)
print(len(out))
for i in range(len(out)):
    # if i%2==1:
    print(out[i])
# out = np.asmatrix(out)
# print(out)
# for i in range(len(out)):
#     print(out[i])
