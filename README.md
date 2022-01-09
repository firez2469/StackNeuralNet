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


### predict() Method
Parameters: NONE

### run() Method
Parameters:
- input data in 2D matrix. Each row being input layer data.

## Dependencies
- Numpy Library

## Requriements
- Python 3 <=

**Note: The library currently is built to only support a binary 1 neuron output layer.**
*The program is based on Jason Brownlee's "How to Code a Neural Network" article and I have expanded upon the python implementation of his code. If you want to 
develop this library further I highly reccomend checking out his tutorial: [Link](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)*
