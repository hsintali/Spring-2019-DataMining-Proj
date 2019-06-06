import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sn

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
        pred = []
        for i in range(0, len(testY)):
            p = self.nearestNeighbor(trainX, trainY, testX[i], param)
            pred.append(p)
            if p == testY[i]:
                correct += 1
        accruacy = float(correct) / float(len(testY)) * 100
        return accruacy, pred


trainX = np.genfromtxt("data/trainX.csv", delimiter=',')
trainY = np.genfromtxt('data/trainY.csv', delimiter=',')

testX = np.genfromtxt("data/testX.csv", delimiter=',')
testY = np.genfromtxt('data/testY.csv', delimiter=',')


"""
kNN = kNearestNeighbor()
testingAccuracy = [0]
maxK = 1
for k in range(1, 30, 2):
    testAcc, pred = kNN.fit(k, trainX, trainY, testX, testY)
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
"""



kNN = kNearestNeighbor()
acc, pred = kNN.fit(17, trainX, trainY, testX, testY)
print("accuracy: {:.2f}".format(acc) + " %")
print(classification_report(testY,pred))
cm = confusion_matrix(testY,pred)
df_cm = pd.DataFrame(cm, ["No" , "Yes"], ["No" , "Yes"])
sn.set(font_scale=1.2)#for label size
sn.heatmap(df_cm, fmt="d", annot=True, annot_kws={"size": 14})# font size
plt.xlabel("Predicted Value", fontsize=14, fontweight='bold')
plt.ylabel("Truth Value", fontsize=14, fontweight='bold')
plt.title("Confusion Matrix of k Nearest Neighbors model", fontsize=14, fontweight='bold')
plt.show()