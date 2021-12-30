# StackNeuralNet
### Description
Basic Supervised Neural Network library on python.

## NeuralNet class

### Constructor
Parameters:
- The input layer size
- The hidden layers size (size of one hidden layer)
- The output layer size

### fit() Method
Parameters:
- training data in 3D vector matrix, first two dimensions being input data for every training sample, and third dimension for the expected output.
- Learning rate.
- The number of iteration the network will learn off of.

### fit2() Method
Parameters:
- training data in 2D Vector matrix. Each row being an input layer for every training sample.
- Expected outputs for each input layer .
- Learning rate.
- The number of iteratins the network will learn off of.

### predict() Method
Parameters: NONE

### run() Method
Parameters:
- input data in 2D matrix. Each row being input layer data.

## Dependencies
- Numpy Library

## Requriements
- Python 3 <=
