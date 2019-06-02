import numpy as np

class CrossValidation:
    def __init__(self, X, Y, folds=3):
        self.folds = folds
        self.splitX, self.splitY = self.splitNFolds(X, Y)

    def splitNFolds(self, X, Y):
        splitX = list()
        splitY = list()
        copyX = list(X)
        copyY = list(Y)
        foldSize = int(len(Y) / self.folds)
        for n in range(self.folds):
            foldX = list()
            foldY = list()
            while len(foldY) < foldSize:
                index = int(np.random.randint(0, len(copyX), 1))
                foldX.append(copyX.pop(index))
                foldY.append(copyY.pop(index))
            splitX.append(foldX)
            splitY.append(foldY)
        return splitX, splitY

    def fit(self, classifier, param, trainX, trainY, validateX, validateY):
        return classifier(param, trainX, trainY, validateX, validateY)

    def exec(self, classifier, param):
        trainAcc = []
        testAcc = []
        for n in range(0, self.folds):
            trainX = []
            trainY = []
            for i in range(0, self.folds):
                if i is not n:
                    trainX.append(self.splitX[i])
                    trainY.append(self.splitY[i])
            validateX = self.splitX[n]
            validateY = self.splitY[n]
            trainX = np.reshape(np.array(trainX), (np.shape(trainX)[0]*np.shape(trainX)[1], np.shape(trainX)[2]))
            trainY = np.reshape(np.array(trainY), (np.shape(trainY)[0]*np.shape(trainY)[1], 1))
            validateX = np.array(validateX)
            validateY = np.array(validateY)
            trainAcc.append(self.fit(classifier, param, trainX, trainY, trainX, trainY))
            testAcc.append(self.fit(classifier, param, trainX, trainY, validateX, validateY))
        return np.array(trainAcc).mean(), np.array(testAcc).mean()