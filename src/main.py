import numpy
import scipy
# from scipy.special import expit as sigmoid
from sigmfunc import sigmoid
import matplotlib.pyplot
import scipy.ndimage

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: sigmoid(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin = 2).T
        targets = numpy.array(targets_list, ndmin = 2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs*(1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs*(1.0 - hidden_outputs)), numpy.transpose(inputs)) 
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin = 2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

data_file = open("mnist_dataset/mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

epochs = 1
for e in range(epochs):
    for record in data_list:
        all_values = record.split(',')
        # image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
        # matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
        # matplotlib.pyplot.show()

        scaled_input = (numpy.asfarray(all_values[1:])/ 255.0*0.99)+0.01
        
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(scaled_input, targets)
        
        # inputs_plus10_img = numpy.asfarray(scipy.ndimage.interpolation.rotate(scaled_input.reshape(28,28),10,cval=0.01, reshape=False))
        # targets = numpy.zeros(output_nodes) + 0.01
        # targets[int(all_values[0])] = 0.99
        # n.train(inputs_plus10_img, targets)

        # inputs_minus10_img = scipy.ndimage.interpolation.rotate(scaled_input.reshape(28,28),-10,cval=0.01, reshape=False)
        # targets = numpy.zeros(output_nodes) + 0.01
        # targets[int(all_values[0])] = 0.99
        # n.train(inputs_minus10_img, targets)
        pass
    pass

test_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_file.readlines()
test_file.close()
scorecard = []
for record in test_data_list:
    all_test_values = record.split(',')
    # print(all_test_values)
    scaled_test_input = (numpy.asfarray(all_test_values[1:])/ 255.0*0.99)+0.01
    final_output = n.query(scaled_test_input)
    # print(final_output)
    network_solution = numpy.argmax(final_output)
    correct_solution = int(all_test_values[0])
    if(network_solution == correct_solution):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = numpy.asarray(scorecard)
print("effect = ", scorecard_array.sum()/scorecard_array.size)