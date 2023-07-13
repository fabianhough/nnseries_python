
from src import NeuralNetwork
from src.neural import NeuralArray, dot_array




if __name__ == '__main__':

    nn = NeuralNetwork(input_size=3, output_size=2, hidden_size_list=[4,5])
    nn.initialize_network()

    print('WEIGHTS:')
    for w in nn.weights:
        print(w)
        print(w.shape)
        print(w.T)
        print(w[0][0])
        print()

    print('BIAS:')
    for b in nn.bias:
        print(b)
        print(b.shape)
        print()


    print('\n\nTESTING')
    sample_input = NeuralArray(3, src_array=[1,2,3])

    sample_output = nn.feed_forward(sample_input)
    print(sample_output)