import numpy as np
import math

import CV
import LR
import kNN


def getCol(A, j):
    return [A_i[j] for A_i in A]

def getCols(A, i, j):
    return [A_i[i:j] for A_i in A]

def mean(x):
    return sum(x) / len(x)

def de_mean(x):
    x_ = mean(x)
    return [x_i - x_ for x_i in x]

def var(x):
    n = len(x)
    deviations = de_mean(x)
    return sum(dev_i * dev_i for dev_i in zip(deviations, deviations)) / (n - 1)

def std(x):
    return math.sqrt(var(x))

def scale(data):
    num_rows = len(data)
    num_cols = len(data[0])
    means = [mean(getCol(data, j)) for j in range(num_cols)]
    stdevs = [std(getCol(data, j)) for j in range(num_cols)]
    return means, stdevs

def normalize(activeDataSet):
    dataSet = activeDataSet
    average = [0.00]*(len(dataSet[0]))
    stds = [0.00]*(len(dataSet[0]))

    for i in dataSet:
        for j in range (0, len(i)):
            average[j] +=  i[j]
    for i in range(len(average)):
        average[i] = (average[i]/len(dataSet))
    for i in dataSet:
        for j in range (0, len(i)):
            stds[j] +=  pow((i[j] - average[j]), 2)
    for i in range(len(stds)):
        stds[i] = math.sqrt(stds[i]/len(dataSet))
    for i in range(len(dataSet)):
        for j in range (0, len(dataSet[0])):
            dataSet[i][j] = (dataSet[i][j] - average[j])/ stds[j]
    return dataSet



dataTrain = np.loadtxt('train.txt')
trainX = getCols(dataTrain, 0, -1)
trainY = getCol(dataTrain, -1)
trainX = normalize(trainX)

dataTest = np.loadtxt('test.txt')
testX = getCols(dataTest, 0, -1)
testY = getCol(dataTest, -1)
testX = normalize(testX)

trainX = np.array(trainX)
trainY = np.reshape(trainY, (len(trainY), 1))
testX = np.array(testX)
testY = np.reshape(testY, (len(testY), 1))

crossValidation = CV.CrossValidation(trainX, trainY, 10)

LR = LR.LogisticRegression()
lmbda = 1.4
print("Cross Validation Accuracy: ", crossValidation.exec(LR.fit, lmbda))
print("Testing Accuracy: ", LR.fit(lmbda, trainX, trainY, testX, testY))

"""
kNN = kNN.kNearestNeighbor()
k = 1
print("Cross Validation Accuracy: ", crossValidation.exec(kNN.fit, k))
print("Testing Accuracy: ", kNN.fit(k, trainX, trainY, testX, testY))
"""