import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp

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
        accuracy = self.accuracyLR(validateX, validateY, w)
        return accuracy * 100

trainX = sp.loadmat("data/trainNorm.mat")["datanorm"][:,0:20]
trainY = sp.loadmat('data/trainY.mat')["result"]
testX = sp.loadmat("data/testNorm.mat")["datanorm"][:,0:20]
testY = sp.loadmat('data/testY.mat')["result"]

trainX = np.array(trainX)
trainY = np.reshape(trainY, (len(trainY), 1))
testX = np.array(testX)
testY = np.reshape(testY, (len(testY), 1))

trainY = np.where(trainY == 0, -1, 1)
testY = np.where(testY == 0, -1, 1)

LR = LogisticRegression()
x = []
testingAccuracy = [0]
bestLambda = 0
lambda_ = 0.1
upper = 100
while lambda_ <= upper:
    x.append(lambda_)
    testAcc = LR.fit(lambda_, trainX, trainY, testX, testY)
    if testAcc > max(testingAccuracy):
        bestLambda = lambda_
    testingAccuracy.append(testAcc)
    print("Testing Accuracy for lambda = {:6.2f}".format(lambda_) + ": {:.2f}".format(testAcc) + " %")
    if (lambda_ == upper):
        break
    lambda_ = lambda_ * 1.5
    if(lambda_ > upper):
        lambda_ = upper

plt.xlim([0.1, upper])
plt.ylim(70, 101)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("$ln\lambda$", fontsize=18, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=18, fontweight='bold')
plt.title("Testing Accuracy for Different $\lambda$", fontsize=20, fontweight='bold')
plt.semilogx(x, testingAccuracy[1:])
plt.show()
print("Logistic Regression with lambda = {:.2f}".format(bestLambda) + " has the highest testing accuracy: {:.2f}".format(max(testingAccuracy)) + " %")