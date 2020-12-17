import functions as f
import numpy as np
from mlxtend.data import loadlocal_mnist
import scipy.sparse.linalg as sc

# using column vectors

# load data
x_train, y_train = loadlocal_mnist(
    images_path='./Data/train-images-idx3-ubyte',
    labels_path='./Data/train-labels-idx1-ubyte'
)

x_test, y_test = loadlocal_mnist(
    images_path='./Data/t10k-images-idx3-ubyte',
    labels_path='./Data/t10k-labels-idx1-ubyte'
)

x_train = x_train.T
x_test = x_test.T

# change y to one-hot encoding
y_train = np.eye(10)[y_train].T
y_test = np.eye(10)[y_test].T

# initialization / parameters
n_x = x_train.shape[0]
n_h = 64

params = {
    "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
    "b1": np.random.randn(n_h, 1) * np.sqrt(1. / n_x),
    "W2": np.random.randn(10, n_h) * np.sqrt(1. / n_h),
    "b2": np.random.randn(10, 1) * np.sqrt(1. / n_h)
}


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
    cache["A2"] = f.sigmoid(cache["Z2"])

    # Z2 = W2.dot(A1) + b2
    cache["Z3"] = np.matmul(params["W2"], cache["A2"]) + params["b2"]

    # A2 = softmax(Z2)
    cache["A3"] = np.exp(cache["Z3"]) / np.sum(np.exp(cache["Z3"]), axis=0)

    return cache


def backprop(X, Y, params, cache, N_batch, lr):
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
    dW2 = (1. / N_batch) * np.matmul(dZ3, cache["A2"].T)
    db2 = (1. / N_batch) * np.sum(dZ3, axis=1, keepdims=True)

    # dZ is only used for calculating other grads -> keep it as matrix
    dZ2 = np.matmul(params["W2"].T, dZ3) * f.dsig(cache["Z2"])
    # hessian of second layer
    dfz = np.zeros((dZ2.shape[0], dZ2.shape[0]))
    for i in range(N_batch):
        dfz_i = f.dsig(cache["Z2"][:, i])
        dfz += np.matmul(dfz_i, dfz_i.T)
    dfz *= (1. / N_batch)
    H2 = (params["W2"].T @ H3 @ params["W2"]) * dfz

    '''
    K2 = np.zeros((dZ3.shape[0]*dZ2.shape[0],dZ3.shape[0]*dZ2.shape[0]))
    for n in range(N_batch):
        a_n = cache["A2"][:, n]
        K2_n = []
        for i in range(H3.shape[0]):
            for j in range(a_n.shape[0]):
                row_ij = a_n[j] * (np.outer(H3[:, i], a_n.T).flatten())
                K2_n.append(row_ij)
        K2 += (1. / N_batch) * np.array(K2_n)
    '''

    a = np.zeros(dZ2.shape[0])
    for n in range(N_batch):
        a_n = cache["A2"][:, n]
        a += a_n * a_n
    a *= (1. / N_batch)
    K2 = [H3*a_i for a_i in a]

    #################################################################################
    # grad of first layer
    dW1 = (1. / N_batch) * np.matmul(dZ2, X.T)
    db1 = (1. / N_batch) * np.sum(dZ2, axis=1, keepdims=True)

    # hessian of first layer
    # H1 = (params["W1"].T @ H2 @ params["W1"])

    '''
    K1 = np.zeros((dZ2.shape[0]*X.shape[0],dZ2.shape[0]*X.shape[0]))
    for n in range(N_batch):
        x_n = X[:, n]
        K1_n = []
        for i in range(H2.shape[0]):
            for j in range(x_n.shape[0]):
                row_ij = x_n[j] * (np.outer(H2[:, i], x_n.T).flatten())
                K1_n.append(row_ij)
        K1 += (1. / N_batch) * np.array(K1_n)
    '''

    '''
    a = np.zeros(X.shape[0])
    for n in range(N_batch):
        a_n = X[:, n]
        a += a_n * a_n
    a *= (1. / N_batch)
    K1 = [H2 * a_i for a_i in a]
    '''
    #################################################################################
    # calculate difference with CG
    '''
    dW2_shape = dW2.shape
    dW2 = dW2.flatten()
    dW2 = sc.cg(np.identity(dW2.shape[0]) - lr * K2, lr * dW2)[0]
    dW2 = dW2.reshape(dW2_shape)
    '''
    
    dW2 = [sc.cg(np.identity(dW2.shape[0])-lr*K2[i],lr*dW2[:,i])[0] for i in range(dW2.shape[1])]
    dW2 = np.array(dW2).T

    db2 = sc.cg(np.identity(db2.shape[0]) - lr * H3, lr * db2)[0]
    db2 = np.array([[i] for i in db2])

    '''
    dW1_shape = dW1.shape
    dW1 = dW1.flatten()
    dW1 = sc.cg(np.identity(dW1.shape[0]) - lr * K1, lr * dW1)
    np.reshape(dW1[0], dW1_shape)
    '''
    '''
    dW1 = [sc.cg(np.identity(dW1.shape[0]) - lr * K1[i], lr * dW1[:, i])[0] for i in range(dW1.shape[1])]
    dW1 = np.array(dW1).T
    '''

    db1 = sc.cg(np.identity(db1.shape[0]) - lr * H2, lr * db1)[0]
    db1 = np.array([[i] for i in db1])

    diffs = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return diffs


# training
epochs = 50
batch_size = 100
learning_rate = 0.2
for i in range(epochs):
    # shuffle dataset
    permutation = np.random.permutation(x_train.shape[1])
    x_train_shuffled = x_train[:, permutation]
    y_train_shuffled = y_train[:, permutation]
    batches = x_train.shape[1] // batch_size

    if i>0 and i%10==0: learning_rate/=2

    for j in range(batches):
        # get mini-batch
        begin = j * batch_size
        end = min(begin + batch_size, x_train.shape[1] - 1)
        X = x_train_shuffled[:, begin:end]
        Y = y_train_shuffled[:, begin:end]
        N_batch = end - begin

        # forward / backward
        cache = forward(X, params)
        diffs = backprop(X, Y, params, cache, N_batch, learning_rate)

        # GD
        params["W1"] = params["W1"] - learning_rate * diffs["dW1"]
        params["b1"] = params["b1"] - learning_rate * diffs["db1"]
        params["W2"] = params["W2"] - learning_rate * diffs["dW2"]
        params["b2"] = params["b2"] - learning_rate * diffs["db2"]

    # calculate loss

    # train loss
    cache = forward(x_train, params)
    train_loss = f.compute_loss(y_train, cache["A3"])

    # test loss
    cache = forward(x_test, params)
    test_loss = f.compute_loss(y_test, cache["A3"])
    print("Epoch {}: training loss = {}, test loss = {}".format(i + 1, train_loss, test_loss))
