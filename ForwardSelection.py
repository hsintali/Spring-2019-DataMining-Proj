import numpy as np
from sklearn.metrics import recall_score

class LogisticRegression:
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def lrloss(self, w, X, Y, lmbda):
        z = Y * np.dot(X, w)
        loss = np.sum(np.log(1 + np.exp(-z))) + 0.5 * lmbda * np.sum(np.power(w[1:], 2))
        return loss

    def lrgrad(self, w, X, Y, lmbda):
        z = Y * np.dot(X, w)
        grad = -np.dot((Y * X).T, (1 - self.sigmoid(z))) + np.multiply(lmbda, w)
        return grad

    def lrhess(self, w, X, Y, lmbda):
        z = Y * np.dot(X, w)
        p = np.reshape(self.sigmoid(z), len(Y))
        D = np.diag(p * (1 - p))
        hess = np.dot(np.dot(X.T, D), X) + lmbda * np.identity(len(w))
        return hess

    def newton(self, w, fn, gradfn, hessfn):
        step = 1.0
        iter = 0
        diff = 1
        loss_old = fn(w)
        while diff > 1e-6:
            iter += 1
            G = gradfn(w)
            H = hessfn(w)
            update = np.linalg.solve(H, G)
            loss_new = fn(w - update)
            if loss_new >= loss_old:
                step *= 2
                update = step * G
                loss_new = fn(w - update)
                while loss_new >= loss_old and step >= 10 ** (-10):
                    step /= 2
                    update = step * G
                    loss_new = fn(w - update)
            diff = loss_old - loss_new
            loss_old = loss_new
            if step < 10 ** (-10):
                break
            w -= update
        return w

    def trainLR(self, X,Y,lmbda):
        w0 = np.zeros((X.shape[1],1))
        return self.newton( w0,
                    lambda w : self.lrloss(w,X,Y,lmbda),
                    lambda w : self.lrgrad(w,X,Y,lmbda),
                    lambda w : self.lrhess(w,X,Y,lmbda))

    def accuracyLR(self, X,Y,w):
        return np.sum(Y*X@w>=0)/Y.shape[0]

    def fit(self, param, trainX, trainY, validateX, validateY):
        w = self.trainLR(trainX, trainY, param)
        Z = validateX @ w
        accuracy = self.accuracyLR(validateX, validateY, w) * 100
        pred = []
        for z in Z:
            if z >= 0:
                pred.append(1)
            else:
                pred.append(-1)
        return accuracy, pred


trainX = np.genfromtxt("data/toyTrainX.csv", delimiter=',')
trainY = np.genfromtxt('data/toyTrainY.csv', delimiter=',')

testX = np.genfromtxt("data/toyTestX.csv", delimiter=',')
testY = np.genfromtxt('data/toyTestY.csv', delimiter=',')

trainX = np.array(trainX)
trainY = np.reshape(trainY, (len(trainY), 1))
testX = np.array(testX)
testY = np.reshape(testY, (len(testY), 1))

trainY = np.where(trainY == 0, -1, 1)
testY = np.where(testY == 0, -1, 1)

def forwardSelection(trainX, trainY, testX, testY):
    selectedFeatures = []
    finalFeatures = []
    finalRecall = 0
    for i in range(1, len(trainX[0]) + 1):
        bestRecall = 0
        pendingFeature = 0
        for j in range(1, len(trainX[0]) + 1):
            if j not in selectedFeatures:
                selectedFeatures.append(j)
                fea = [item - 1 for item in selectedFeatures]
                LR = LogisticRegression()
                acc, pred = LR.fit(0.76, trainX[:, fea], trainY, testX[:, fea], testY)
                recall = recall_score(testY, pred) * 100
                print("        Using feature(s) " + str(selectedFeatures) + " recall is " + str(recall) + "%")
                selectedFeatures.remove(j)
                if recall > bestRecall:
                    bestRecall = recall
                    pendingFeature = j
        if pendingFeature == 0:
            break
        elif pendingFeature != 0:
            selectedFeatures.append(pendingFeature)
        if bestRecall > finalRecall:
            finalRecall = bestRecall
            finalFeatures = selectedFeatures[:]
            print("\nFeature set " + str(selectedFeatures) + " was best, recall is " + str(bestRecall) + "%\n")
        else:
            print("\n(Warning, Recall has decreased! Continuing search in case of local maxima)")
            print("Feature set " + str(selectedFeatures) + " was best, recall is " + str(bestRecall) + "%\n")
    print("\nFinished search!! The best feature subset is " + str(finalFeatures) + ", which has an recall of " + str(finalRecall) + "%\n")

forwardSelection(trainX, trainY, testX, testY)