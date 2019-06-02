import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp
import CV
import kNN

trainX = sp.loadmat("trainNorm.mat")["datanorm"][:,0:20]
trainY = sp.loadmat('trainY.mat')["result"]
testX = sp.loadmat("testNorm.mat")["datanorm"][:,0:20]
testY = sp.loadmat('testY.mat')["result"]

crossValidation = CV.CrossValidation(trainX, trainY, 10)
kNN = kNN.kNearestNeighbor()

validationAccuracy = [0]
maxK = 1
for k in range(1, 30, 2):
    acc = crossValidation.exec(kNN.fit, k)
    if acc > max(validationAccuracy):
        maxK = k
    validationAccuracy.append(acc)
    print("Cross Validation Accuracy for k = {:2d}".format(k) + ": {:.4f}".format(acc))

plt.xlim(1, 29)
plt.ylim(1, 105)
plt.xticks(range(1, 30, 2), fontsize=16)
plt.yticks(fontsize=16)
plt.title("Cross Validation Accuracy for Different k", fontsize=20, fontweight='bold')
plt.xlabel("k", fontsize=18, fontweight='bold')
plt.ylabel("Cross Validation Accuracy (%)", fontsize=18, fontweight='bold')
plt.plot(range(1, 30, 2), validationAccuracy[1:])
plt.show()

print("KNN with k = " + str(maxK) + " will have the highest accuracy: {:.4f}".format(max(validationAccuracy)) + "%")
print("Testing Accuracy (k = " + str(maxK) + "): {:.4f}".format(kNN.fit(k, trainX, trainY, testX, testY)) + "%")