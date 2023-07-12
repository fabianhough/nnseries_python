
from src import NeuralNetwork




if __name__ == '__main__':

    nn = NeuralNetwork(input_size=3, output_size=2, hidden_size_list=[4,5])
    nn.initialize_network()

    print('WEIGHTS:')
    for w in nn.weights:
        print(w)
        print()

    print('BIAS:')
    for b in nn.bias:
        print(b)
        print()