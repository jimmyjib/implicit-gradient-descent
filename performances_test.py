import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import dataExtraction as de
import gradientDescent as gd
import gradientDescent_test as gd_t

MAXITER = 20000
LRATE = 0.01
REG = 0.001
TTIME = 10
SHUFFLED = True

ttime = range(0, MAXITER, TTIME)

train, trainLabels, test, testLabels = de.ionosphere(0.2)

print("TESTS : {} iterations, learning rate {}".format(MAXITER, LRATE))

# print("GRADIENT DESCENT")
# weight, trainG, testG = gd.gradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME, shuffled = SHUFFLED, regularization = REG)

print("IMPLICIT GRADIENT DESCENT")
weight, trainI, testI = gd_t.implicitGradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME, shuffled = SHUFFLED, regularization = REG)

# print("STOCHASTIC IMPLICIT GRADIENT DESCENT")
# weight, trainSI, testSI = gd_t.stochasticImplicitGradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME, shuffled = SHUFFLED, regularization = REG)

# plt.figure(1)

# #Training error
# plt.subplot(211)
# plt.plot(ttime, trainG, color="blue", linewidth=1.0, linestyle="-", label="GD")
# plt.plot(ttime, trainI, color="red", linewidth=1.0, linestyle="-", label="IGD")
# # plt.plot(ttime, trainSI, color="green", linewidth=1.0, linestyle="-", label="SIGD")

# plt.xlim(0, MAXITER)
# plt.xlabel('Number of iterations')
# plt.ylabel('Error')
# plt.title('Training error')

# # log
# plt.subplot(212)
# plt.plot(ttime, testG, color="blue", linewidth=1.0, linestyle="-", label="GD")
# plt.plot(ttime, testI, color="red", linewidth=1.0, linestyle="-", label="IGD")
# # plt.plot(ttime, testSI, color="green", linewidth=1.0, linestyle="-", label="SIGD")

# plt.xlim(0, MAXITER)
# plt.xlabel('Number of iterations')
# plt.ylabel('Error')
# plt.title('Testing error')

# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# plt.show()

