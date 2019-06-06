from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from itertools import product
import scipy.io as scio
import numpy as np 

trainfilename = 'datanorm-full.mat'
trainmat = scio.loadmat(trainfilename)
traindata = np.asarray(trainmat['datanorm'])
print(traindata)

# testfilename = 'datanorm.mat'
# testmat = scio.loadmat(testfilename)
# testdata = np.asarray(testmat['datanorm'])
# testX = testdata[:, 0:20]
# testY = testdata[:, 20]

trainX = traindata[0:40000, 0:20]
trainY = traindata[0:40000, 20]

testX = traindata[40000:41188, 0:20]
testY = traindata[40000:41188, 20]


# training with different 

# parameters
# default (100, )
my_hidden_layer_sizes = (3, )
# ('identity', 'logistic', 'tanh', 'relu'), default relu
activations = ['identity', 'logistic', 'tanh', 'relu']
my_activation = 'relu'
# {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
solvers = ['lbfgs', 'sgd', 'adam']
my_solver = 'adam'
# default 0.0001, l2 penalty
my_alpha = 0.0001
# default ‘auto’, related to slover
my_batch_size = 'auto'
# learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
learning_rates = ['constant', 'invscaling', 'adaptive']
my_learning_rate = 'constant'
# learning_rate_init : double, optional, default 0.001, only for sgd and adam solver
my_learning_rate_init = 0.001
# default False
my_early_stopping = False
# default 200
my_max_iter = 200

for my_activation in activations:
	for my_solver in solvers:
	# my_hidden_layer_sizes = (i, )
		clf = MLPClassifier(hidden_layer_sizes=my_hidden_layer_sizes, alpha=my_alpha, 
			activation=my_activation, solver=my_solver,
			learning_rate=my_learning_rate, learning_rate_init=my_learning_rate_init,
			batch_size=my_batch_size, max_iter=my_max_iter, 
			random_state=1)
		clf.fit(trainX, trainY)
		preY = clf.predict(testX)
		accu = accuracy_score(preY, testY)
		print("predicting done, the parameters are: \n my_hidden_layer_sizes: ", my_hidden_layer_sizes, 
			"  my_activation: ", my_activation,
			"  my_solver: ", my_solver)
		print("accuracy", accu)


