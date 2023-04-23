import numpy as np

def neural_network(input, weights):
    size = weights.shape
    input_size = input.shape
    # print(input_size)
    # print(size)
    if size[1] != input.shape[0]:
        print("Wrong rows/cols number")
        return 1

    out = weights * input

    return out


if __name__ == '__main__':
    weights = np.matrix('0.1 0.1 -0.3;0.1 0.2 0.0;0.0 0.7 0.1;0.2 0.4 0.0;-0.3 0.5 0.1')
    input = np.matrix('0.5;0.75;0.1')
    out = neural_network(input,weights)
    print(out)