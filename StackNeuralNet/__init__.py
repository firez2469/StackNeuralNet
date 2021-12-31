import random
import math

from random import random


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def __activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
    return activation

def transfer(activation):
    try:
        val = 1.0/(1.0+math.exp(-activation))
    except OverflowError:
        val = 1
    return val

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = __activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

#calculate the derivative of a neuron's output
def transfer_derivative(output):
    return output*(1.0-output)

#Backpropagate error and store it in the neurons
def backward_propagate_error(network,expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error+=(neuron['weights'][j]*neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j]*transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
          inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate*neuron['delta']*inputs[j]
            neuron['weights'][-1] -= l_rate*neuron['delta']
def train_network(network,train,l_rate,n_epoch,n_outputs,debug=False):
    for epoch in range(n_epoch):
        sum_error=  0
        for row in train:
            outputs = forward_propagate(network,row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]]=1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network,expected)
            update_weights(network,row,l_rate)
        if debug:
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network,row):
    outputs = forward_propagate(network,row)
    return outputs.index(max(outputs))



class NeuralNet:
    def __init__(self,n_inputs,n_hidden,n_outputs):
        self.network = initialize_network(n_inputs,n_hidden,n_outputs)

        self.n_outputs = n_outputs
    def fit(self,train,l_rate,n_epoch):
        self.train=train
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        train_network(self.network,train,l_rate,n_epoch,self.n_outputs)
    def fit2(self,train,expect,l_rate,n_epoch):
        for row in range(len(expect)):
            newRow = []
            for val in train[row]:
                newRow.append(val)
            newRow.append(expect[row])

            train[row]=newRow
            #train[row].append(expect[row])
        self.l_rate=l_rate
        self.train=train
        self.n_epoch=n_epoch

    def predict(self):
        totalError = 0
        for row in self.train:
            prediction = predict(self.network, row)
            print('Expected=%d, Got=%d' % (row[-1], prediction))
            totalError+=(prediction-row[-1])**2
        print('Final Error:',totalError)
    def run(self,data):
        out = []
        for row in data:
            prediction = predict(self.network,row)
            out.append(prediction)
        return out




#additional Util functions

#takes in a value, the min value, and max value and makes array from min to max all 0 except for expected output
def convertToArrayData(value, min,max):
    arr= []
    for val in range(max-min):
        v = val+min
        if v == value:
            arr.append(1)
        else:
            arr.append(0)
