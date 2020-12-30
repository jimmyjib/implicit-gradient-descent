import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import functions as f 
import numpy as np
from mlxtend.data import loadlocal_mnist
import scipy.sparse.linalg as sc

import gradientDescent as gd
import pdb

#parameter
EPOCH = 100
BATCHSIZE = 200
LRATE = 0.2
TTIME = 10
ttime = range(EPOCH // TTIME)


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

#initialization
IGD = {
    "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
    "b1": np.random.randn(n_h, 1) * np.sqrt(1. / n_x),
    "W2": np.random.randn(10, n_h) * np.sqrt(1. / n_h),
    "b2": np.random.randn(10, 1) * np.sqrt(1. / n_h)
}

SGD = {
    "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
    "b1": np.random.randn(n_h, 1) * np.sqrt(1. / n_x),
    "W2": np.random.randn(10, n_h) * np.sqrt(1. / n_h),
    "b2": np.random.randn(10, 1) * np.sqrt(1. / n_h)
}

ADAM = {
    "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
    "b1": np.random.randn(n_h, 1) * np.sqrt(1. / n_x),
    "W2": np.random.randn(10, n_h) * np.sqrt(1. / n_h),
    "b2": np.random.randn(10, 1) * np.sqrt(1. / n_h)
}

print("#############################################################\n")
print("TESTS : {} iterations, learning rate {}".format(EPOCH, LRATE))
print("\n")
print("#############################################################\n")
print("STOCASTIC GRADIENT DESCENT")
_, trainSGD, testSGD = gd.cal_SGD(x_train, y_train, x_test, y_test, SGD, batch_size=BATCHSIZE, epochs=EPOCH, learning_rate=LRATE, testTime=TTIME)
print("#############################################################\n")
print("ADAM GRADIENT DESCENT")
_, trainADAM, testADAM = gd.cal_ADAM(x_train, y_train, x_test, y_test, ADAM, batch_size=BATCHSIZE, epochs=EPOCH, learning_rate=LRATE, testTime=TTIME)
print("#############################################################\n")
print("IMPLICIT GRADIENT DESCENT")
_, trainIGD, testIGD = gd.cal_IGD(x_train, y_train, x_test, y_test, IGD, batch_size=BATCHSIZE, epochs=EPOCH, learning_rate=LRATE, testTime=TTIME)

plt.figure(1)

#trainig error
plt.subplot(211)
plt.plot(ttime, trainSGD, color="blue", linewidth="1.3", linestyle="-", label="SGD")
plt.plot(ttime, trainADAM, color="black", linewidth="1.3", linestyle="-", label="ADAM")
plt.plot(ttime, trainIGD, color="red", linewidth="1.7", linestyle="-", label="IGD")

plt.xlim(0, EPOCH // TTIME)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Training error')

#test error
plt.subplot(212)
plt.plot(ttime, testSGD, color="blue", linewidth="1.3", linestyle="-", label="SGD")
plt.plot(ttime, testADAM, color="black", linewidth="1.3", linestyle="-", label="ADAM")
plt.plot(ttime, testIGD, color="red", linewidth="1.7", linestyle="-", label="IGD")

plt.xlim(0, EPOCH // TTIME)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Testing error')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()