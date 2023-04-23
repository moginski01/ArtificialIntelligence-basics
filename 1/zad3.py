import numpy as np


def deep_neural_network(input_vector, scales):
    # print("test")
    for i in range(len(scales)):
        input_shape = input_vector.shape
        matrix_shape = scales[i].shape

        if matrix_shape[1] == input_shape[0] or matrix_shape[0] == input_shape[0]:
            input_vector = scales[i] * input_vector
        else:
            print("Wrong size of scales or input_vector")
            return -1;

    return input_vector


a = np.matrix('0.1 0.1 -0.3;0.1 0.2 0.0;0.0 0.7 0.1;0.2 0.4 0.0;-0.3 0.5 0.1')
b = np.matrix('0.7 0.9 -0.4 0.8 0.1;0.8 0.5 0.3 0.1 0.0;-0.3 0.9 0.3 0.1 -0.2')

inp = np.matrix('0.5; 0.75; 0.1')

scales = []
scales.append(a)
scales.append(b)

# print(len(scales))
out = deep_neural_network(inp, scales)
print(out)