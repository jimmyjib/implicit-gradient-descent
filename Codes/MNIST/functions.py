import numpy as np
import scipy.sparse.linalg as sc


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


def forward(X, params):
    '''
    2-layer FC network

    :param X: input data
    :param params: dictionary of weight and bias
    :return: dictionary of z(pre-activation), a(activation)
    '''
    cache = {}

    # Z1 = W1.dot(x) + b1
    cache["Z2"] = np.matmul(params["W1"], X) + params["b1"]

    # A1 = sigmoid(Z1)
    cache["A2"] = sigmoid(cache["Z2"])

    # Z2 = W2.dot(A1) + b2
    cache["Z3"] = np.matmul(params["W2"], cache["A2"]) + params["b2"]

    # A2 = softmax(Z2)
    cache["A3"] = np.exp(cache["Z3"]) / np.sum(np.exp(cache["Z3"]), axis=0)

    return cache

def backprop_IGD(X, Y, params, cache, N_batch, lr):
    """
    :param X: input data
    :param Y: labels
    :param params: weight parameters
    :param cache: forward pass values
    :param N_batch: number of datas per batch

    :return: dictionary of difference
    """
    #################################################################################
    # grad of last layer
    dZ3 = cache["A3"] - Y
    # hessian of last layer
    H3 = np.zeros((dZ3.shape[0], dZ3.shape[0]))
    for i in range(N_batch):
        H3 += np.matmul(dZ3[:, i], dZ3[:, i].T)
    H3 *= (1. / N_batch)

    #################################################################################
    # grad of second layer
    # dZ is only used for calculating other grads -> keep it as matrix
    dZ2 = np.matmul(params["W2"].T, dZ3) * dsig(cache["Z2"])
    # hessian of second layer
    dfz = np.zeros((dZ2.shape[0], dZ2.shape[0]))
    for i in range(N_batch):
        dfz_i = dsig(cache["Z2"][:, i])
        dfz += np.matmul(dfz_i, dfz_i.T)
    dfz *= (1. / N_batch)
    H2 = (params["W2"].T @ H3 @ params["W2"]) * dfz

    #################################################################################
    # grad of first layer
    dW1 = (1. / N_batch) * np.matmul(dZ2, X.T)
    dW2 = (1. / N_batch) * np.matmul(dZ3, cache["A2"].T)
    db1 = (1. / N_batch) * np.sum(dZ2, axis=1, keepdims=True)
    db2 = (1. / N_batch) * np.sum(dZ3, axis=1, keepdims=True)

    a = np.zeros(dZ2.shape[0])
    for n in range(N_batch):
        a_n = cache["A2"][:, n]
        a += a_n * a_n
    a *= (1. / N_batch)
    K2 = [H3*a_i for a_i in a]
    dW2 = [sc.cg(np.identity(dW2.shape[0])-lr*K2[i],lr*dW2[:,i])[0] for i in range(dW2.shape[1])]
    dW2 = np.array(dW2).T


    #################################################################################
    # calculate difference with CG

    db2 = sc.cg(np.identity(db2.shape[0]) - lr * H3, lr * db2)[0]
    db2 = np.array([[i] for i in db2])


    db1 = sc.cg(np.identity(db1.shape[0]) - lr * H2, lr * db1)[0]
    db1 = np.array([[i] for i in db1])

    diffs = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return diffs

def backprop_origin(X, Y, params, cache, N_batch, lr) : 
    """
    params X : input data
    params Y : labels
    param params : weight and bias parameter 
    param cache : forward pass values
    param N_batch: number of datas per batch 

    return : dictionary of difference
    """
    ###grad of last layer 
    dZ3 = cache["A3"] - Y

    ###grad of second layer 
    dW2 = (1. / N_batch) * np.matmul(dZ3, cache["A2"].T)
    db2 = (1. / N_batch) * np.sum(dZ3, axis=1, keepdims=True) 

    dZ2 = np.matmul(params["W2"].T, dZ3) * dsig(cache["Z2"])

    ###grad of first layer
    dW1 = (1. / N_batch) * np.matmul(dZ2, X.T) 
    db1 = (1. / N_batch) * np.sum(dZ2, axis=1, keepdims=True)

    ###make dictionary
    diffs = {"dW1" : dW1, "db1" : db1, "dW2" : dW2, "db2" : db2} 

    return diffs

