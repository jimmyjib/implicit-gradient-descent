import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import dataExtraction as de
import gradientDescent as gd

MAXITER = 1000
LRATE = 0.01
REG = 0.001
TTIME = 10
SHUFFLED = True

ttime = range(0, MAXITER, TTIME)

train, trainLabels, test, testLabels = de.ionosphere(0.2)

print("TESTS : {} iterations, learning rate {}".format(MAXITER, LRATE))
print("GRADIENT DESCENT")
weight, trainG, testG = gd.gradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME, shuffled = SHUFFLED, regularization = REG)
print("STOCHASTIC GRADIENT DESCENT")
weight, trainS, testS = gd.stochasticGradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME, regularization = REG)
print("BATCH GRADIENT DESCENT")
weight, trainB, testB = gd.batchGradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME, shuffled = SHUFFLED, regularization = REG)
print("ADAM GRADIENT DESCENT")
weight, trainA, testA = gd.adamGradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME, shuffled = SHUFFLED, regularization = REG)
print("EVE GRADIENT DESCENT")
weight, trainE, testE = gd.eveGradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME, shuffled = SHUFFLED, regularization = REG)
print("ADAM BATCH GRADIENT DESCENT")
weight, trainBA, testBA = gd.adamBatchGradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME, shuffled = SHUFFLED, regularization = REG)
print("EVE BATCH GRADIENT DESCENT")
weight, trainBE, testBE = gd.eveBatchGradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME, shuffled = SHUFFLED, regularization = REG)

plt.figure(1)

# Training error
plt.subplot(211)
plt.plot(ttime, trainG, color="blue", linewidth=1.0, linestyle="-", label="GD")
plt.plot(ttime, trainS, color="red", linewidth=1.0, linestyle="-", label="SGD")
plt.plot(ttime, trainB, color="yellow", linewidth=1.0, linestyle="--", label="BGD")
plt.plot(ttime, trainA, color="black", linewidth=1.0, linestyle="-", label="Adam")
plt.plot(ttime, trainE, color="green", linewidth=1.0, linestyle="-", label="Eve")
plt.plot(ttime, trainBA, color="magenta", linewidth=1.0, linestyle="--", label="AdamB")
plt.plot(ttime, trainBE, color="black", linewidth=1.0, linestyle="--", label="EveB")

plt.xlim(0, MAXITER)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Training error')

# log
plt.subplot(212)
plt.plot(ttime, testG, color="blue", linewidth=1.0, linestyle="-", label="GD")
plt.plot(ttime, testS, color="red", linewidth=1.0, linestyle="-", label="SGD")
plt.plot(ttime, testB, color="yellow", linewidth=1.0, linestyle="--", label="BGD")
plt.plot(ttime, testA, color="black", linewidth=1.0, linestyle="-", label="Adam")
plt.plot(ttime, testE, color="green", linewidth=1.0, linestyle="-", label="Eve")
plt.plot(ttime, testBA, color="magenta", linewidth=1.0, linestyle="--", label="AdamB")
plt.plot(ttime, testBE, color="black", linewidth=1.0, linestyle="--", label="EveB")

plt.xlim(0, MAXITER)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Testing error')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
