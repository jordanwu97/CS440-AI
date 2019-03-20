import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

class MultiClassPerceptron(object):
	def __init__(self,num_class,feature_dim):
		"""Initialize a multi class perceptron model. 

		This function will initialize a feature_dim weight vector,
		for each class. 

		The LAST index of feature_dim is assumed to be the bias term,
			self.w[:,0] = [w1,w2,w3...,BIAS] 
			where wi corresponds to each feature dimension,
			0 corresponds to class 0.  

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example 
		"""

		self.w = np.zeros((feature_dim+1,num_class))

	def train(self,train_set,train_label):
		""" Train perceptron model (self.w) with training dataset. 

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""

		num_examples = train_set.shape[0]
		num_dimensions = train_set.shape[1]
		num_class = self.w.shape[1]
		
		# onehot matrix for given y
		y = np.zeros((train_label.shape[0], self.w.shape[1]))
		y[np.arange(y.shape[0]), train_label] = 2
		y -= 1

		# add biasing term for each example
		train_set_biased = np.c_[train_set, np.ones(train_set.shape[0])]

		learn_rate = 0.5

		# calculate yhat
		for epoch in range(200):
			print (epoch)
			yhat = np.sign(np.matmul(train_set_biased, self.w))
			self.w += np.matmul(np.transpose(train_set_biased),y - yhat) * learn_rate

	def test(self,test_set,test_label):
		""" Test the trained perceptron model (self.w) using testing dataset. 
			The accuracy is computed as the average of correctness 
			by comparing between predicted label and true label. 
			
		Args:
		    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
		    test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

		Returns:
			accuracy(float): average accuracy value 
			pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
		"""    

		# YOUR CODE HERE
		pred_label = np.zeros((len(test_set)))

		test_set_biased = np.c_[test_set, np.ones(test_set.shape[0])]
		yhat = np.matmul(test_set_biased,self.w)
		
		pred_label = np.argmax(yhat, axis=1)

		accuracy = np.sum(np.equal(test_label,pred_label)) / len(test_set)

		print ("accuracy:", accuracy)
		
		return accuracy, pred_label

	def save_model(self, weight_file):
		""" Save the trained model parameters 
		""" 

		np.save(weight_file,self.w)
 
	def load_model(self, weight_file):
		""" Load the trained model parameters 
		""" 

		self.w = np.load(weight_file)

