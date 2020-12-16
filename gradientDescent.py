import pandas as pd
import numpy as np
import numpy.random as rand
import pdb

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

def gradientDescent(train, trainLabels, test, testLabels,
	maxIter = 10, learningRate = 0.001, regularization = 0.01,
	shuffled = True, testTime = 2):
	"""
	Computes the gradient descent in order to predict the labels
	-> Binary classification by logistic regression
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

		# Computes the gradient on all the training data
		for j in range(len(train)):
			grad += logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)/len(train)

		# Updates the weight with a regularizer in order to avoid overfitting
		weight -= learningRate*(grad + regularization*weight)

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

	return weight, lossesTrain, lossesTest

def stochasticGradientDescent(train, trainLabels, test, testLabels,
	maxIter = 10, learningRate = 0.001, regularization = 0.01, testTime = 2):
	"""
	Computes the stochastic gradient descent in order to predict labels
	-> Binary classification
	"""
	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))

	# Multiplication is only for an easier comparison with other algorithms
	for i in range(maxIter * len(train)):
		# Takes a random sample
		j = rand.randint(len(train))

		# Computes the local direction
		grad = logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)

		# Updates the weight
		weight -= learningRate*(grad + regularization*weight)/len(train)

		# Computes the error on the training and testing sets
		if (i % (testTime*len(train)) == 0):
			print("Iteration : {} / {}".format(i+1, maxIter*len(train)))
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

	return weight, lossesTrain, lossesTest


def batchGradientDescent(train, trainLabels, test, testLabels,
	maxIter = 10, learningRate = 0.001, regularization = 0.01,
	batchSizePercentage = 0.1, shuffled = True, testTime = 2):
	"""
	Computes the gradient descent in order to predict the labels, updates after a batch
	-> Binary classification by logistic regression
	"""
	batchSize = int(len(train)*batchSizePercentage)
	numberOfBatch = int(len(train)/batchSize)
	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))
	for i in range(maxIter):
		grad = np.zeros(weight.shape)

		# Shuffles the data in order to improve the learning
		if shuffled:
			train, trainLabels = shuffle(train, trainLabels)

		# A batch is a subset of the total training set
		for batch in range(numberOfBatch):
			for j in range(batch*batchSize, (batch+1)*batchSize):
				grad += logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)/len(train)

			# Updates the weight with the subset direction
			# Division is in order to compute the same gradient than gradient descent at the end
			weight -= learningRate*(grad + regularization*weight/numberOfBatch)

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

	return weight, lossesTrain, lossesTest

def adamGradientDescent(train, trainLabels, test, testLabels,
	maxIter = 10, learningRate = 0.001, regularization = 0.01,
	shuffled = True, testTime = 2,
	b1 = 0.9, b2 = 0.999, epsilon = 10**(-8)):
	"""
	Computes the gradient descent in order to predict labels thanks to the eve algorithm
	-> Binary classification
	"""
	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))

	# Moving averages
	m = np.zeros(weight.shape) # Mean
	v = np.zeros(weight.shape) # Variance
	# To compute the moving average
	b1t = 1
	b2t = 1

	for i in range(maxIter):
		if i < 1000:
			b1t *= b1
			b2t *= b2
		else:
			b1t = 0
			b2t = 0
		grad = np.zeros(weight.shape)

		# Shuffles the data in order to improve the learning
		if shuffled:
			train, trainLabels = shuffle(train, trainLabels)

		# Computes the full gradient
		for j in range(len(train)):
			grad += logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)/len(train)

		# Updates the moving averages
		m = b1*m + (1-b1)*grad
		mh = m / (1-b1t)

		v = b2*v + (1-b2)*np.multiply(grad,grad)
		vh = v/(1-b2t)

		# Updates the weight with the normalized gradient
		weight -= learningRate*(np.multiply(mh,1/(np.sqrt(vh) + epsilon)) + regularization*weight)

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

	return weight, lossesTrain, lossesTest

def eveGradientDescent(train, trainLabels, test, testLabels,
	maxIter = 10, learningRate = 0.001, regularization = 0.01,
	shuffled = True, testTime = 2,
	b1 = 0.9, b2 = 0.999, b3 = 0.999, epsilon = 10**(-8), k = 0.1, K = 10):
	"""
	Computes the gradient descent in order to predict labels thanks to the eve algorithm
	-> Binary classification
	"""
	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))

	# Moving averages
	m = np.zeros(weight.shape)
	v = np.zeros(weight.shape)
	# To compute them
	b1t = 1
	b2t = 1
	# Adaptative learning rate
	d = 1
	# Loss of the last epoch
	oldLoss = 0

	for i in range(maxIter):
		if i < 1000:
			b1t *= b1
			b2t *= b2
		else:
			b1t = 0
			b2t = 0
		loss = 0
		grad = np.zeros(weight.shape)

		# Shuffles the data in order to improve the learning
		if shuffled:
			train, trainLabels = shuffle(train, trainLabels)

		# Computes the full gradient and error
		for j in range(len(train)):
			grad += logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)/len(train)
			loss += logisticLoss(train.iloc[j], trainLabels.iloc[j], weight)/len(train)

		# Updates the moving averages
		m = b1*m + (1-b1)*grad
		mh = m / (1-b1t)

		v = b2*v + (1-b2)*np.multiply(grad,grad)
		vh = v/(1-b2t)

		# Updates the adaptative learning rate
		if (i > 0):
			# In order to bound the learning rate
			if loss >= oldLoss:
				delta = k + 1
				Delta = K + 1
			else:
				delta = 1/(K+1)
				Delta = 1/(k+1)
			c = min(max(delta, loss/oldLoss), Delta)
			oldLossS = oldLoss
			oldLoss = c*oldLoss
			# Computes the feedback of the error function (normalized)
			r = abs(oldLoss - oldLossS)/(min(oldLoss,oldLossS))
			# Updates the correction of learning rate
			d = b3*d + (1-b3)*r
		else:
			oldLoss = loss

		# Updates the weight
		weight -= learningRate*(np.multiply(mh,1/(d*np.sqrt(vh) + epsilon)) + regularization*weight)

		# Computes the error on the training and testing sets
		if (i % testTime == 0):
			print("Iteration : {} / {}".format(i+1, maxIter))
			print("\t-> Train Loss : {}".format(loss))
			lossesTrain.append(loss)
			loss = 0
			for j in range(len(test)):
				loss += logisticLoss(test.iloc[j], testLabels.iloc[j], weight)/len(test)
			lossesTest.append(loss)
			print("\t-> Test Loss : {}".format(loss))

	return weight, lossesTrain, lossesTest

def adamBatchGradientDescent(train, trainLabels, test, testLabels,
	maxIter = 10, learningRate = 0.001, regularization = 0.01,
	batchSizePercentage = 0.1, shuffled = True, testTime = 2,
	b1 = 0.9, b2 = 0.999, epsilon = 10**(-8)):
	"""
	Computes the gradient descent in order to predict labels thanks to the eve algorithm
	-> Binary classification
	"""
	batchSize = int(len(train)*batchSizePercentage)
	numberOfBatch = int(len(train)/batchSize)
	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))

	# Moving averages
	m = np.zeros(weight.shape)
	v = np.zeros(weight.shape)
	# To compute them
	b1t = 1
	b2t = 1

	for i in range(maxIter):
		# Shuffles the data in order to improve the learning
		if shuffled:
			train, trainLabels = shuffle(train, trainLabels)

		for batch in range(numberOfBatch):
			if i*batch < 1000:
				b1t *= b1
				b2t *= b2
			else:
				b1t = 0
				b2t = 0
			grad = np.zeros(weight.shape)

			# Computes the gradient on the batch
			for j in range(batch*batchSize, (batch+1)*batchSize):
				grad += logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)/len(train)

			# Computes local moving averages
			m = b1*m + (1-b1)*grad
			mh = m / (1-b1t)

			v = b2*v + (1-b2)*np.multiply(grad,grad)
			vh = v/(1-b2t)

			weight -= learningRate*(np.multiply(mh,1/(np.sqrt(vh) + epsilon)) + regularization*weight/numberOfBatch)

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

	return weight, lossesTrain, lossesTest

def eveBatchGradientDescent(train, trainLabels, test, testLabels,
	maxIter = 10, learningRate = 0.001, regularization = 0.01,
	batchSizePercentage = 0.1, shuffled = True, testTime = 2,
	b1 = 0.9, b2 = 0.999, b3 = 0.999, epsilon = 10**(-8), k = 0.1, K = 10):
	"""
	Computes the gradient descent in order to predict labels thanks to the eve algorithm
	-> Binary classification
	"""
	batchSize = int(len(train)*batchSizePercentage)
	numberOfBatch = int(len(train)/batchSize)
	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))

	m = np.zeros(weight.shape)
	v = np.zeros(weight.shape)
	d = 1
	oldLoss = 0
	b1t = 1
	b2t = 1

	for i in range(maxIter):
		# Shuffles the data in order to improve the learning
		if shuffled:
			train, trainLabels = shuffle(train, trainLabels)

		# Computes error on full training set
		for batch in range(numberOfBatch):
			if i*batch < 1000:
				b1t *= b1
				b2t *= b2
			else:
				b1t = 0
				b2t = 0
			loss = 0
			grad = np.zeros(weight.shape)

			# Computes the gradient on a batch
			for j in range(batch*batchSize, (batch+1)*batchSize):
				grad += logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)/len(train)
				loss += logisticLoss(train.iloc[j], trainLabels.iloc[j], weight)/len(train)

			# Computes a sub approximation of the moving averages
			m = b1*m + (1-b1)*grad
			mh = m / (1-b1t)

			v = b2*v + (1-b2)*np.multiply(grad,grad)
			vh = v/(1-b2t)

			# Updates the learning rate (see Eve explanation)
			if (i + batch > 0):
				if loss >= oldLoss:
					delta = k + 1
					Delta = K + 1
				else:
					delta = 1/(K+1)
					Delta = 1/(k+1)
				c = min(max(delta, loss/oldLoss), Delta)
				oldLossS = oldLoss
				oldLoss = c*oldLoss
				r = abs(oldLoss - oldLossS)/(min(oldLoss,oldLossS))
				d = b3*d + (1-b3)*r
			else:
				oldLoss = loss

			# Updates weight
			weight -= learningRate*(np.multiply(mh,1/(np.sqrt(vh) + epsilon)) + regularization*weight/numberOfBatch)

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

	return weight, lossesTrain, lossesTest


