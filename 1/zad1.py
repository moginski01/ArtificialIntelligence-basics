import numpy as np

def neuron(input_vector, weights, bias):
    weights_shape = weights.shape
    for i in range(weights_shape[0]):
        out = weights[i] * input_vector
        out = out.sum(dtype=float)
        print(out)


if __name__ == '__main__':
    weights = np.matrix('0.1 0.1 -0.3;0.1 0.2 0.0;0.0 0.7 0.1;0.2 0.4 0.0;-0.3 0.5 0.1')
    input = np.matrix('0.5;0.75;0.1')

    out = neuron(input, weights, 0)
    # print(out)


    