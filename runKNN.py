import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp
import CV
import kNN

trainX = sp.loadmat("data/trainNorm.mat")["datanorm"][:,0:20]
trainY = sp.loadmat('data/trainY.mat')["result"]
testX = sp.loadmat("data/testNorm.mat")["datanorm"][:,0:20]
testY = sp.loadmat('data/testY.mat')["result"]

crossValidation = CV.CrossValidation(trainX, trainY, 10)
kNN = kNN.kNearestNeighbor()
trainingAccuracy = [0]
validationAccuracy = [0]
maxK = 1
for k in range(1, 14, 2):
    trainAcc, validationAcc = crossValidation.exec(kNN.fit, k)
    if validationAcc > max(validationAccuracy):
        maxK = k
    trainingAccuracy.append(trainAcc)
    validationAccuracy.append(validationAcc)
    print("Cross Validation Accuracy for k = {:2d}".format(k) + ": {:.2f}".format(validationAcc) + " %")

plt.xlim(1, 13)
plt.ylim(70, 101)
plt.xticks(range(1, 14, 2), fontsize=16)
plt.yticks(range(70, 101, 5), fontsize=16)
plt.title("Cross Validation Accuracy for Different k", fontsize=20, fontweight='bold')
plt.xlabel("k", fontsize=18, fontweight='bold')
plt.ylabel("Cross Validation Accuracy (%)", fontsize=18, fontweight='bold')
plt.plot(range(1, 14, 2), trainingAccuracy[1:], color='r', label="Training Accuracy")
plt.plot(range(1, 14, 2), validationAccuracy[1:], color='b', label="Validation Accuracy")
plt.legend(loc='best', fontsize=16)
plt.show()

print("KNN with k = " + str(maxK) + " will have the highest accuracy: {:.2f}".format(max(validationAccuracy)) + " %")
print("Testing Accuracy (k = " + str(maxK) + "): {:.2f}".format(kNN.fit(k, trainX, trainY, testX, testY)) + " %")