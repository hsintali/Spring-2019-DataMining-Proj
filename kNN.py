import numpy as np

class kNearestNeighbor:
    def nearestNeighbor(self, trainSetX, trainSetY, testX, k):
        distance = (((testX - trainSetX)**2).sum(axis=1))**0.5
        distance = dict(zip(distance, trainSetY))
        candidate = [[key, distance[key]] for key in sorted(distance.keys())[:k]]
        result = np.zeros(2)
        for element in candidate:
            result[int(element[1])] += 1
        return np.argmax(result)

    def fit(self, param, trainX, trainY, validateX, validateY):
        correct = 0
        for i in range(0, len(validateY)):
            p = self.nearestNeighbor(trainX, trainY, validateX[i], param)
            if p == validateY[i]:
                correct += 1
        return float(correct) / float(len(validateY)) * 100