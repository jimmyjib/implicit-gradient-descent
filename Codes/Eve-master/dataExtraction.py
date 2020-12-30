import os
import pandas
import urllib.request
import numpy as np

def trainAndTest(data, testPercentage):
	"""
	Shuffles the data and separates in two datasets
	"""
	permutation = np.random.permutation(data.index)
	indice = int(testPercentage*len(permutation))
	train = data.loc[permutation[indice:]]
	test = data.loc[permutation[:indice]]
	return train, test

def binarization(labels, value):
	"""
	Changes the labels by checking if they are greater or lower than a given value
	"""
	return labels.apply(lambda x : -1 if x != value else 1)


def download(url, fileName, saveDirectory):
	"""
	Downloads the given fileName
	"""
	if not(os.path.exists(saveDirectory + fileName)):
		response = urllib.request.urlopen(url + fileName)

		with open(saveDirectory + fileName, 'wb') as out:
		    out.write(response.read())

		print("Success")
	else :
		print("Data already downloaded")

def abalone(testPercentage, dataset = "Data/abalone.data"):
	"""
	Reads the abalone dataset and returns train and test subdatasets
	"""
	data = pandas.read_csv(dataset, header=None)
	mapping = {'M':-1, 'F':1, 'I':0}
	data = data.replace({data.columns[0]:mapping})

	train, test = trainAndTest(data, testPercentage)
	features = list(range(0,len(data.columns)-2))

	return train[features], train.iloc[:,-1], test[features], test.iloc[:,-1]

def ionosphere(testPercentage, dataset = "Data/ionosphere.data"):
	"""
	Reads the ionosphere dataset and returns train and test subdatasets
	"""
	data = pandas.read_csv(dataset, header=None)
	mapping = {'g':-1, 'b':1}
	data = data.replace({data.columns[len(data.columns)-1]:mapping})

	train, test = trainAndTest(data, testPercentage)
	features = list(range(0,len(data.columns)-2))

	return train[features], train.iloc[:,-1], test[features], test.iloc[:,-1]

if __name__ == '__main__':
	print("Download")
	download("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/", "abalone.data", "Data/")
	download("https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/", "ionosphere.data", "Data/")
