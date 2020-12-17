import numpy as np


# Encode onehot matrix from vector of class labels
def encode_onehot(labels):
    return (np.arange(np.max(labels) + 1) == labels) * 1.0


# Make prediction classes
def make_pred_class(y):
    return np.argmax(y, axis=1)
