import random
import math


### ARRAY METHODS/CLASSES ###

def build_array(dim1, dim2):
    new_array = []
    for i in range(dim1):
        new_row = []
        for j in range(dim2):
            new_row.append(random.random() * 2 - 1) # Setting value between (-1, 1)
        new_array.append(new_row)
    return new_array

class NeuralArray():
    def __init__(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2
        self.array = build_array(dim1, dim2)

    @property
    def shape(self):
        return (self.dim1, self.dim2)

    def __repr__(self):
        repr_str = '['
        for i in range(self.dim1):
            repr_str += str(self.array[i])
            if i < self.dim1-1:
                repr_str += '\n'
        repr_str += ']'
        return repr_str

### END ARRAY ###


### ACTIVATION FUNCTIONS ###

def relu(x):
    return max(0, x)

def relu_deriv(x):
    return 1 if x > 0

def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x))

def sigmoid_deriv(x):
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)

### END ACTIVATION FUNCTIONS


### NEURAL NETWORK ###

class NeuralNetwork():
    def __init__(self, input_size: int, output_size: int, hidden_size_list: list):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list

        self.weights = []
        self.bias = []
    
    def initialize_network(self):
        net_dims = [self.input_size] + self.hidden_size_list + [self.output_size]

        for i in range(len(net_dims)-1):
            layer_dims = (net_dims[i], net_dims[i+1])
            self.weights.append(NeuralArray(layer_dims[0], layer_dims[1]))
            self.bias.append(NeuralArray(layer_dims[1], 1))


### END NEURAL NETWORK