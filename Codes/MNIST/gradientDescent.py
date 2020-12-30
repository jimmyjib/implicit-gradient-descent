import functions as f 
import numpy as np
import pdb


def cal_SGD (x_train, y_train, x_test, y_test, SGD, batch_size, epochs, learning_rate, testTime) : 
    train_loss = []
    test_loss = []

    for i in range(epochs) :
        #shuffle dataset
        permutation = np.random.permutation(x_train.shape[1]) 
        x_train_shuffled = x_train[:, permutation]
        y_train_shuffled = y_train[:, permutation]
        batches = x_train.shape[1] // batch_size

        if i>0 and i%(epochs/2)==0 : learning_rate /= 2

        for j in range(batches) : 
            #get mini batch
            begin = j * batch_size
            end = min(begin + batch_size, x_train.shape[1] - 1)
            X = x_train_shuffled[:, begin:end]
            Y = y_train_shuffled[:, begin:end] 
            N_batch = end - begin

            #forward & backward
            cache_sgd = f.forward(X, SGD)
            diffs_sgd = f.backprop_origin(X, Y, SGD, cache_sgd, N_batch, learning_rate)

            #calculate SGD
            SGD["W1"] = SGD["W1"] - learning_rate * diffs_sgd["dW1"]
            SGD["b1"] = SGD["b1"] - learning_rate * diffs_sgd["db1"]
            SGD["W2"] = SGD["W2"] - learning_rate * diffs_sgd["dW2"]
            SGD["b2"] = SGD["b2"] - learning_rate * diffs_sgd["db2"]
        if( i % testTime == 0) : 
            print("ITERATION : {} / {}".format(i+1, epochs))
            #train loss
            loss = 0
            cache_sgd = f.forward(x_train, SGD)
            loss = f.compute_loss(y_train, cache_sgd["A3"])
            train_loss.append(loss)
            print("\t-> Train Loss : {}".format(loss))

            #test loss 
            loss = 0
            cache_sgd = f.forward(x_test, SGD) 
            loss = f.compute_loss(y_test, cache_sgd["A3"])
            test_loss.append(loss)
            print("\t-> Test Loss : {}".format(loss))

    return SGD, train_loss, test_loss


def cal_ADAM (x_train, y_train, x_test, y_test, ADAM, batch_size, epochs, learning_rate, testTime, regularization = 0.01, b1 = 0.9, b2 = 0.999, epslion = 10**(-8)):
    train_loss = []
    test_loss = []
    #moving averages
    mW1 = np.zeros(ADAM["W1"].shape) 
    vW1 = np.zeros(ADAM["W1"].shape)

    mb1 = np.zeros(ADAM["b1"].shape)
    vb1 = np.zeros(ADAM["b1"].shape)

    mW2 = np.zeros(ADAM["W2"].shape)
    vW2 = np.zeros(ADAM["W2"].shape)

    mb2 = np.zeros(ADAM["b2"].shape)
    vb2 = np.zeros(ADAM["b2"].shape)

    #compute moving average
    b1t = 1
    b2t = 1

    for i in range(epochs) :

        #change parameter
        b1t *= b1
        b2t *= b2
        

        #shuffle dataset
        permutation = np.random.permutation(x_train.shape[1]) 
        x_train_shuffled = x_train[:, permutation]
        y_train_shuffled = y_train[:, permutation]
        batches = x_train.shape[1] // batch_size

        if i>0 and i%(epochs/2)==0 : learning_rate /= 2

        for j in range(batches) : 
            #get mini batch
            begin = j * batch_size
            end = min(begin + batch_size, x_train.shape[1] - 1)
            X = x_train_shuffled[:, begin:end]
            Y = y_train_shuffled[:, begin:end] 
            N_batch = end - begin

            #forward & backward
            cache_adam = f.forward(X, ADAM)
            diffs_adam = f.backprop_origin(X, Y, ADAM, cache_adam, N_batch, learning_rate)

            #update moving averages
            mW1 = b1 * mW1 + (1 - b1) * diffs_adam["dW1"]
            mW1h = mW1 / (1 - b1t)
            vW1 = b2 * vW1 + (1 - b2) * np.multiply(diffs_adam["dW1"] ,diffs_adam["dW1"])
            vW1h = vW1 / (1 - b2t)

            mb1 = b1 * mb1 + (1 - b1) * diffs_adam["db1"]
            mb1h = mb1 / (1 - b1t)
            vb1 = b2 * vb1 + (1 - b2) * np.multiply(diffs_adam["db1"] ,diffs_adam["db1"])
            vb1h = vb1 / (1 - b2t)
            
            mW2 = b1 * mW2 + (1 - b1) * diffs_adam["dW2"]
            mW2h = mW2 / (1 - b1t)
            vW2 = b2 * vW2 + (1 - b2) * np.multiply(diffs_adam["dW2"] ,diffs_adam["dW2"])
            vW2h = vW2 / (1 - b2t)

            mb2 = b1 * mb2 + (1 - b1) * diffs_adam["db2"]
            mb2h = mb2 / (1 - b1t)
            vb2 = b2 * vb2 + (1 - b2) * np.multiply(diffs_adam["db2"] ,diffs_adam["db2"])
            vb2h = vb2 / (1 - b2t)
            
            #calculate ADAM
            ADAM["W1"] = ADAM["W1"] - learning_rate * (np.multiply(mW1h, 1. / (np.sqrt(vW1h) + epslion)) + regularization * ADAM["W1"])
            ADAM["b1"] = ADAM["b1"] - learning_rate * (np.multiply(mb1h, 1. / (np.sqrt(vb1h) + epslion)) + regularization * ADAM["b1"])
            ADAM["W2"] = ADAM["W2"] - learning_rate * (np.multiply(mW2h, 1. / (np.sqrt(vW2h) + epslion)) + regularization * ADAM["W2"])
            ADAM["b2"] = ADAM["b2"] - learning_rate * (np.multiply(mb2h, 1. / (np.sqrt(vb2h) + epslion)) + regularization * ADAM["b2"])
        
        if( i % testTime == 0) : 
            print("ITERATION : {} / {}".format(i+1, epochs))
            #train loss
            loss = 0
            cache_adam = f.forward(x_train, ADAM)
            loss = f.compute_loss(y_train, cache_adam["A3"])
            train_loss.append(loss)
            print("\t-> Train Loss : {}".format(loss))

            #test loss 
            loss = 0
            cache_adam = f.forward(x_test, ADAM) 
            loss = f.compute_loss(y_test, cache_adam["A3"])
            test_loss.append(loss)
            print("\t-> Test Loss : {}".format(loss))

    return ADAM, train_loss, test_loss

def cal_IGD (x_train, y_train, x_test, y_test, IGD, batch_size, epochs, learning_rate, testTime):
    train_loss = []
    test_loss = []

    for i in range(epochs):
        # shuffle dataset
        permutation = np.random.permutation(x_train.shape[1])
        x_train_shuffled = x_train[:, permutation]
        y_train_shuffled = y_train[:, permutation]
        batches = x_train.shape[1] // batch_size

        if i>0 and i%(epochs/2)==0: learning_rate/=2

        for j in range(batches):
            # get mini-batch
            begin = j * batch_size
            end = min(begin + batch_size, x_train.shape[1] - 1)
            X = x_train_shuffled[:, begin:end]
            Y = y_train_shuffled[:, begin:end]
            N_batch = end - begin

            # forward / backward / loss
            loss = 0
            cache = f.forward(X, IGD)
            diffs = f.backprop_IGD(X, Y, IGD, cache, N_batch, learning_rate)
            loss = f.compute_loss(Y, cache["A3"])

            # GD
            IGD["W1"] = IGD["W1"] - learning_rate * diffs["dW1"]
            IGD["b1"] = IGD["b1"] - learning_rate * diffs["db1"]
            IGD["W2"] = IGD["W2"] - learning_rate * diffs["dW2"]
            IGD["b2"] = IGD["b2"] - learning_rate * diffs["db2"]

        if( i % testTime == 0) : 
            # calculate loss
            print("ITERATION : {} / {}".format(i+1, epochs))
            # train loss
            loss = 0
            cache = f.forward(x_train, IGD)
            loss = f.compute_loss(y_train, cache["A3"])
            train_loss.append(loss)
            print("\t-> Train Loss : {}".format(loss))

            # test loss
            loss = 0
            cache = f.forward(x_test, IGD)
            loss = f.compute_loss(y_test, cache["A3"])
            test_loss.append(loss)
            print("\t-> Test Loss : {}".format(loss))

    return IGD, test_loss, train_loss







