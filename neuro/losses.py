import numpy as np
from neuro.misc.defaults import EPSILON


# Aggregate softmax cross entropy loss function
def softmax_cross_entropy(y, labels):
    return - np.sum(labels * np.log(y + EPSILON))


# Aggregate softmax cross entropy loss function derivative
def softmax_cross_entropy_derivative(y, labels):
    return y - labels


# Aggregate binary cross entropy loss function
def binary_cross_entropy(y, labels):
    bce = - np.dot(labels.T, np.log(y + EPSILON)) - np.dot((1 - labels).T, np.log(1 - y + EPSILON))
    return bce[0][0]


# Aggregate binary cross entropy loss function derivative
def binary_cross_entropy_derivative(y, labels):
    return y - labels
