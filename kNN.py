import numpy as np
import math

class kNearestNeighbor:
    def euclidean(self, trainSet, testSet):
        dist = 0
        for i in range(0, len(trainSet)):
            dist += (trainSet[i] - testSet[i])**2
        return math.sqrt(dist)

    def nearestNeighbor(self, trainX, trainY, testX):
        distance = []
        for i in range(0, len(trainX)):
            dist = self.euclidean(trainX[i], testX)
            distance.append([dist, trainY[i]])
        distance = sorted(distance)
        return distance[0][1]

    def fit(self, param, trainX, trainY, validateX, validateY):
        correct = 0
        for i in range(0, len(validateY)):
            p = self.nearestNeighbor(trainX, trainY, validateX[i])
            if p == validateY[i]:
                correct += 1
        return float(correct) / float(len(validateY)) * 100