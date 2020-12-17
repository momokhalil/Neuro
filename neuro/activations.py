import numpy as np


# leaky-ReLU
def l_relu(x):
    return x * (x > 0) + 0.1 * x * (x <= 0)


# leaky-ReLU Derivative
def d_l_relu(x):
    return 1 * (x > 0) + 0.1 * (x <= 0)


# ReLU
def relu(x):
    return x * (x > 0)


# ReLU Derivative
def d_relu(x):
    return 1 * (x > 0)


# Tanh
def tanh(x):
    return np.tanh(x)


# Tanh derivative
def d_tanh(x):
    return 1 - np.tanh(x) ** 2


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Sigmoid function
def d_sigmoid(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


# Pass-through function
def linear(z):
    return z


# Pass-through function derivative
def d_linear(z):
    return 1


# Softmax function
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e, axis=1, keepdims=True)
