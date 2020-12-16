import pandas as pd
import numpy as np
import numpy.random as rand
import pdb
import scipy.sparse.linalg as sc

from autograd import elementwise_grad as egrad
from autograd import jacobian
import autograd.numpy as anp

def shuffle(data, label):
    """
    Shuffles the data and associated labels
    """
    permutation = np.random.permutation(data.index)
    return data.loc[permutation], label.loc[permutation]

def logisticLoss(features, label, weight):
    """
    Computes the logistic loss
    """
    return np.log(1 + np.exp(-label*features.dot(weight)))

    
def logisticGrad(features, label, weight):
    """
    Computes the gradient of the logistic loss
    """
    denum = 1 +  np.exp(label*features.dot(weight))
    return np.multiply(features,-label/denum)


def hessian(features, label, weight): 
    """
    Calculate the hessian matrix with finite differences
    Parameters:
        - features : ndarray, [281, 33]
        - label    : ndarray, [281]
        - weight   : ndarray, [33]
    Returns:
        an array of shape [33, 33]
        where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    hess = np.zeros((len(weight), len(weight))) # [33, 33]
    const = 0
    for k in range(len(features)):
        val = np.exp(-label[k]*features[k].dot(weight))
        const += label[k] * (val/(1 + val)**2)
    const = const/len(features)

    for i in range(len(weight)):
        for j in range(len(weight)):
            hess[i][j] = const * weight[j] * (weight[i] **2)
            
    return hess

#def conjugate_grad(A, b, x=None):
#    """
#    Description
#    -----------
#    Solve a linear equation Ax = b with conjugate gradient method.
#    Parameters
#    ----------
#    A: 2d numpy.array of positive semi-definite (symmetric) matrix
#    b: 1d numpy.array
#    x: 1d numpy.array of initial point
#    Returns
#    -------
#    1d numpy.array x such that Ax = b
#    """
#    n = len(b)
#    if not x:
#        x = np.ones(n)
#    r = np.dot(A, x) - b
#    p = - r
#    r_k_norm = np.dot(r, r)
#    for i in range(2*n):
#        Ap = np.dot(A, p)
#        alpha = r_k_norm / np.dot(p, Ap)
#        x += alpha * p
#        r += alpha * Ap
#        r_kplus1_norm = np.dot(r, r)
#        beta = r_kplus1_norm / r_k_norm
#        r_k_norm = r_kplus1_norm
#        if r_kplus1_norm < 1e-5:
#            #print 'Itr:', i
#            break
#        p = beta * p - r
#    return x


def implicitGrad(features, label, weight, learningRate):
    """
    Compute the gradient of the implicit 
        Parameters
        ----------
        features: 2d numpy.array 
        label: 1d numpy.array
        weight: 1d numpy.array 
        Returns
        -------
        1d numpy.array dx such that the solution of Adx = b 
    """

    # Calculate cost function
    # Computes the gradient on all the training data
    grad = np.zeros(weight.shape)
    for j in range(len(features)):
        grad += logisticGrad(features[j], label[j], weight)/len(features)
    # shape : [1]

    # Calculate hess function
    hess = hessian(features, label, weight)
    I = np.identity(len(weight))

    # Calculate conjugate_grad
    A = I - hess
    #dx = conjugate_grad(A, learningRate * grad)
    dx = sc.cg(A, learningRate * grad)
    

    return dx

def implicitGradientDescent(train, trainLabels, test, testLabels, maxIter = 10, learningRate = 0.001, regularization = 0.01, shuffled = True, testTime = 2):
    """
    Computes the implicit gradient descent in order to predict labels
    -> Binary classification
    """
    lossesTest = []
    lossesTrain = [] 
    weight = np.zeros(len(train.columns))
    for i in range(maxIter):
        loss = 0
        grad = np.zeros(weight.shape)
        
        # Shuffles the data in order to improve the learning
        if shuffled: 
            train, trainLabels = shuffle(train, trainLabels)
            features = pd.DataFrame.to_numpy(train)
            label = pd.DataFrame.to_numpy(trainLabels)
            
        # Calculate implicit gradient descent on all training data
        dx = implicitGrad(features, label, weight, learningRate)
        
        # Updates the weight 
        dx = np.asarray(dx)
        weight -= dx[0]
        
        # Computes the error on the training and testing sets
        if (i % testTime == 0):
            print("Iteration : {} / {}".format(i+1, maxIter))
            loss = 0
            for j in range(len(train)):
                loss += logisticLoss(train.iloc[j], trainLabels.iloc[j], weight)/len(train)
            lossesTrain.append(loss)
            print("\t-> Train Loss : {}".format(loss))
            loss = 0
            for j in range(len(test)):
                loss += logisticLoss(test.iloc[j], testLabels.iloc[j], weight)/len(test)
            lossesTest.append(loss)
            print("\t-> Test Loss : {}".format(loss))
            
        if (i == 5000) :
            learningRate = learningRate / 2
        elif (i ==  10000) : 
            learningRate = learningRate / 2
        elif (i == 50000) : 
            learningRate = learningRate / 2
        

    return weight, lossesTrain, lossesTest


# def stochasticImplicitGradientDescent(train, trainLabels, test, testLabels,
# 	maxIter = 10, learningRate = 0.001, regularization = 0.01, testTime = 2):

