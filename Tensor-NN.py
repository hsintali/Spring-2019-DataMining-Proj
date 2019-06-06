from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

class NeuralNetwork:
	def __init__(self, trainNameX, trainNameY, threshold):
		self.trainNameX = trainNameX
		self.trainNameY = trainNameY
		self.threshold = threshold
		self.loadFile()

	def runNN(self, fea, title):
		self.title = title
		self.selectFeatures(fea)
		self.fit()

	def loadFile(self):
		self.trainX_full = np.loadtxt(self.trainNameX, delimiter=',')
		self.trainY = np.loadtxt(self.trainNameY, delimiter=',')

		self.testX_full = np.loadtxt("testX.csv", delimiter=',')
		self.testY = np.loadtxt("testY.csv", delimiter=',')

	def selectFeatures(self, fea):
		self.trainX = np.take(self.trainX_full, fea, axis=1)
		self.testX = np.take(self.testX_full, fea, axis=1)
		print(self.trainX.shape)
		self.input_dim = len(fea)

	def fit(self):
		print ("Neural Network for", self.title, " : ")

		# Neural Nerwork Models configuration
		# np.random.seed(1)
		model = Sequential()
		model.add(Dense(32, input_dim=self.input_dim, activation='relu'))
		# model.add(Dense(64, activation='relu'))
		# model.add(Dropout(0, 5))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		# training
		model.fit(self.trainX, self.trainY, epochs=10, batch_size=32)
		# predicting
		print(self.title)
		model.evaluate(self.testX, self.testY, batch_size=32)
		self.resultY = model.predict(self.testX)
		# threshold = 0.5
		self.classifier()
		self.ConfusionMatrix()

		ones_predict = np.count_nonzero(self.predictY)
		ones_test = np.count_nonzero(self.testY)
		print("test one: ", ones_test, " predict one: ", ones_predict)
		print("Neural Networks done for", self.title, " !!")
		
	def ConfusionMatrix (self):
		CM = {
		'True_Positive' : 0,
		'True_Negative' : 0,
		'False_Positive' : 0,
		'False_Negative' : 0
		}
		self.RR = {
		'recall' : 0,
		'precision' : 0,
		'recall_no' : 0,
		'precision_no' : 0
		}
		for i in range (self.testY.shape[0]):
			if self.predictY[i] == 1:
				if self.predictY[i] == self.testY[i] :
					CM['True_Positive'] += 1
				else :
					CM['False_Positive'] += 1
			else :
				if self.predictY[i] == self.testY[i] :
					CM['True_Negative'] += 1
				else :
					CM['False_Negative'] += 1
		self.RR['recall'] = CM['True_Positive'] / (CM['True_Positive'] + CM['False_Negative'])
		self.RR['precision'] = CM['True_Positive'] / (CM['True_Positive'] + CM['False_Positive'])
		self.RR['recall_no'] = CM['True_Negative'] / (CM['True_Negative'] + CM['False_Positive'])
		self.RR['precision_no'] = CM['True_Negative'] / (CM['True_Negative'] + CM['False_Negative'])
		
		# print("Confusion Matrix: \n", self.CM)
		print(self.title)
		print("Recognition Rate: \n", self.RR)

	def classifier (self):
		self.predictY = np.zeros(self.resultY.shape[0])
		for i in range (self.resultY.shape[0]) :
			if self.resultY[i] > self.threshold :
				self.predictY[i] = 1

# all features
fea_all = np.arange(20)
# features selected by PCA
fea_PCA = [0, 1, 2, 4, 5, 6, 7, 9, 11, 12]
# features selected by forward selection
fea_FS = [4, 1, 5, 9, 6, 8, 11, 15, 10, 18, 13, 0, 7, 19]

threshold = 0.5

# trainNameX, trainNameY, threshold
NN = NeuralNetwork("trainX.csv", "trainY.csv", threshold)
NN.runNN(fea_all, "all features and no oversampling")
NN.runNN(fea_PCA, "PCA and no oversampling")
NN.runNN(fea_FS, "forward selection and no oversampling")

NN_O = NeuralNetwork("trainX_oversampling.csv", "trainY_oversampling.csv", threshold)
NN_O.runNN(fea_all, "all features and oversampling")
NN_O.runNN(fea_PCA, "PCA and oversampling")
NN_O.runNN(fea_FS, "forward selection and oversampling")










