import numpy as np


def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s

def dsig(z):
    ds = sigmoid(z)*(1-sigmoid(z))
    return ds

def compute_loss(Y, Y_hat):
    '''
    cross entropy loss
    '''
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1];
    L = -(1. / m) * L_sum

    return L

