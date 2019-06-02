import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp

class kNearestNeighbor:
    def nearestNeighbor(self, trainSetX, trainSetY, testX, k):
        distance = (((testX - trainSetX)**2).sum(axis=1))**0.5
        distance = dict(zip(distance, trainSetY))
        candidate = [[key, distance[key]] for key in sorted(distance.keys())[:k]]
        result = np.zeros(2)
        for element in candidate:
            result[int(element[1])] += 1
        return np.argmax(result)

    def fit(self, param, trainX, trainY, testX, testY):
        correct = 0
        for i in range(0, len(testY)):
            p = self.nearestNeighbor(trainX, trainY, testX[i], param)
            if p == testY[i]:
                correct += 1
        return float(correct) / float(len(testY)) * 100

trainX = sp.loadmat("data/trainNorm.mat")["datanorm"][:,0:20]
trainY = sp.loadmat('data/trainY.mat')["result"]
testX = sp.loadmat("data/testNorm.mat")["datanorm"][:,0:20]
testY = sp.loadmat('data/testY.mat')["result"]

kNN = kNearestNeighbor()
testingAccuracy = [0]
maxK = 1
for k in range(1, 30, 2):
    testAcc = kNN.fit(k, trainX, trainY, testX, testY)
    if testAcc > max(testingAccuracy):
        maxK = k
    testingAccuracy.append(testAcc)
    print("Testing Accuracy for k = {:2d}".format(k) + ": {:.2f}".format(testAcc) + " %")

plt.xlim(1, 29)
plt.ylim(70, 101)
plt.xticks(range(1, 30, 2), fontsize=16)
plt.yticks(range(70, 101, 5), fontsize=16)
plt.title("Testing Accuracy for Different k", fontsize=20, fontweight='bold')
plt.xlabel("k", fontsize=18, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=18, fontweight='bold')
plt.plot(range(1, 30, 2), testingAccuracy[1:])
plt.show()
print("KNN with k = " + str(maxK) + " has the highest testing accuracy: {:.2f}".format(max(testingAccuracy)) + " %")