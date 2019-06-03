import scipy.io as scio
import numpy as np

trainfilename = 'datanorm-full.mat'
trainmat = scio.loadmat(trainfilename)
traindata = np.asarray(trainmat['datanorm'])
size = traindata.shape[0]
test_size = int(size/10)

sample = np.random.choice(size, size=test_size, replace=False)
train_XY = np.delete(traindata, sample, axis=0)
test_XY = traindata[sample, :]

np.savetxt("trainXY.csv", train_XY, delimiter=",")
np.savetxt("testXY.csv", test_XY, delimiter=",")
trainX = train_XY[:, 0:20]
trainY = train_XY[:, 20]
np.savetxt("trainX.csv", trainX, delimiter=",")
np.savetxt("trainY.csv", trainY, delimiter=",")
print(np.count_nonzero(trainY))

testX = test_XY[:, 0:20]
testY = test_XY[:, 20]
np.savetxt("testX.csv", testX, delimiter=",")
np.savetxt("testY.csv", testY, delimiter=",")
print(np.count_nonzero(testY))
