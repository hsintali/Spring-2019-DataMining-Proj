from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np


def ConfusionMatrix (y_true, y_pred):
	CM = {
	'True_Positive' : 0,
	'True_Negative' : 0,
	'False_Positive' : 0,
	'False_Negative' : 0
	}
	RR = {
	'recall' : 0,
	'specificity' : 0,
	'precision' : 0
	}
	for i in range (y_true.shape[0]):
		if y_pred[i] == 1:
			if y_pred[i] == y_true[i] :
				CM['True_Positive'] += 1
			else :
				CM['False_Positive'] += 1
		else :
			if y_pred[i] == y_true[i] :
				CM['True_Negative'] += 1
			else :
				CM['False_Negative'] += 1
	RR['recall'] = CM['True_Positive'] / (CM['True_Positive'] + CM['False_Negative'])
	RR['specificity'] = CM['True_Negative'] / (CM['False_Negative'] + CM['True_Negative'])
	RR['precision'] = CM['True_Positive'] / (CM['True_Positive'] + CM['False_Positive'])
	print("Confusion Matrix: \n", CM)
	print("Recognition Rate: \n", RR)

def classifier (threshold, resultY):
	predictY = np.zeros(resultY.shape[0])
	for i in range (resultY.shape[0]) :
		if resultY[i] > threshold :
			predictY[i] = 1
	return predictY


trainX = np.loadtxt("trainX.csv", delimiter=',')
trainY = np.loadtxt("trainY.csv", delimiter=',')
testX = np.loadtxt("testX.csv", delimiter=',')
testY = np.loadtxt("testY.csv", delimiter=',')

# features selected by PCA
fea_PCA = [0, 1, 2, 4, 5, 6, 7, 9, 11, 12]
# features selected by forward selection
fea_FS = [4, 1, 5, 9, 6, 8, 11, 15, 10, 18, 13, 0, 7, 19]

# np.random.seed(1)
model = Sequential()
model.add(Dense(32, input_dim=20, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0, 5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=10, batch_size=32)
score = model.evaluate(testX, testY, batch_size=32)
resultY = model.predict(testX)
threshold = 0.5
predictY = classifier(threshold, resultY)
ones_predict = np.count_nonzero(predictY)
ones_test = np.count_nonzero(testY)
print("test one: ", ones_test, " predict one: ", ones_predict)
ConfusionMatrix(testY, predictY)





