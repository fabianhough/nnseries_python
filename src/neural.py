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

def build_flat_array(dim):
    return [random.random() * 2 - 1 for i in range(dim)]


def dot_array(flat_array, dim_array):
    assert flat_array.shape[0] == dim_array.shape[0]
    result = []
    for arr in dim_array.T:
        result.append(sum([flat_array[i]*arr[i] for i in range(len(flat_array))]))

    return NeuralArray(dim1=len(result), src_array=result)

def add_array(flat_array1, flat_array2):
    assert len(flat_array1) == len(flat_array2)
    return NeuralArray(
        dim1=len(flat_array1),
        src_array=[flat_array1[i]+flat_array2[i] for i in range(len(flat_array1))]
    )

def subtract_array(flat_array1, flat_array2):
    assert len(flat_array1) == len(flat_array2)
    return NeuralArray(
        dim1=len(flat_array1),
        src_array=[flat_array1[i]-flat_array2[i] for i in range(len(flat_array1))]
    )

class NeuralArray():
    def __init__(self, dim1: int, dim2: int=None, src_array: list=None):
        self.dim1 = dim1
        self.dim2 = dim2
        if src_array is None:
            if dim2 is None:
                self.array = build_flat_array(dim1)
            else:
                self.array = build_array(dim1, dim2)
        else:
            self.array = src_array

    @property
    def shape(self):
        return (self.dim1, self.dim2)

    def __getitem__(self, idx):
        if idx >= self.dim1:
            raise IndexError(f'{idx} Larger Than {self.dim1}')
        return self.array[idx]

    def __repr__(self):
        repr_str = '['
        for i in range(self.dim1):
            repr_str += str(self.array[i])
            if i < self.dim1-1:
                repr_str += '\n'
        repr_str += ']'
        return repr_str

    def __len__(self):
        return len(self.array)

    @property
    def T(self):
        assert not self.dim2 is None
        T_array = []
        for j in range(self.dim2):
            T_row = []
            for i in range(self.dim1):
                T_row.append(self.array[i][j])
            T_array.append(T_row)
        return NeuralArray(self.dim2, self.dim1, T_array)

    def apply_func(self, func):
        if self.dim2 is None:
            return NeuralArray(dim1=self.dim1, src_array=[func(a) for a in self.array])
        else:
            new_array = []
            for i in range(self.dim1):
                new_array.append([func(a) for a in self.array[i]])
            return NeuralArray(dim1=self.dim1, dim2=self.dim2, src_array=new_array)



### END ARRAY ###


### ACTIVATION FUNCTIONS ###

def relu(x):
    return max(0, x)

def relu_deriv(x):
    if x > 0:
        return 1
    return 0

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

        self.layers = {}
    
    def initialize_network(self):
        net_dims = [self.input_size] + self.hidden_size_list + [self.output_size]

        for i in range(len(net_dims)-1):
            layer_dims = (net_dims[i], net_dims[i+1])
            self.weights.append(NeuralArray(layer_dims[0], layer_dims[1]))
            self.bias.append(NeuralArray(layer_dims[1]))

            self.layers[i] = [1 for j in range(net_dims[i+1]+1)]

    def feed_forward(self, input_array):
        assert len(input_array) == self.input_size

        self.layers[0] = input_array
        for i in range(len(self.weights)):
            result = add_array(dot_array(self.layers[i], self.weights[i]), self.bias[i])
            if i < len(self.weights) - 1:
                result = result.apply_func(relu)
            else:
                result = result.apply_func(sigmoid)

            self.layers[i+1] = result

        return self.layers[len(self.layers.keys())-1]

        


### END NEURAL NETWORK