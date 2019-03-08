import numpy as np

class NaiveBayes(object):
	def __init__(self,num_class,feature_dim,num_value):
		"""Initialize a naive bayes model. 

		This function will initialize prior and likelihood, where 
		prior is P(class) with a dimension of (# of class,)
			that estimates the empirical frequencies of different classes in the training set.
		likelihood is P(F_i = f | class) with a dimension of 
			(# of features/pixels per image, # of possible values per pixel, # of class),
			that computes the probability of every pixel location i being value f for every class label.  

		Args:
		    num_class(int): number of classes to classify 10
		    feature_dim(int): feature dimension for each example 784
		    num_value(int): number of possible values for each pixel 256
		"""

		self.num_value = num_value
		self.num_class = num_class
		self.feature_dim = feature_dim

		self.prior = np.zeros((num_class))
		self.likelihood = np.zeros((feature_dim,num_value,num_class))

	def train(self,train_set,train_label):
		""" Train naive bayes model (self.prior and self.likelihood) with training dataset. 
			self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,) 10
			self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of 
				(# of features/pixels per image, # of possible values per pixel, # of class). 784 x 256 x 10
			You should apply Laplace smoothing to compute the likelihood. 

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim) 50000 x 784
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, ) 50000
		"""
		# Count priors
		label, count = np.unique(train_label, return_counts=True) 
		for l,c in zip(label,count):
			self.prior[l] = int(c)

		# Sort axis
		label_sort_axis = np.argsort(train_label)
		arranged_train_set = train_set[label_sort_axis,:]

		# Rearrange train_set using sorted axis
		base = 0
		for label, count in zip(range(self.num_class),self.prior):
			# cut a batch with same label out from arranged_train_set
			batch = np.transpose(arranged_train_set[int(base):int(base + count)])
			# for each pixel in the batch...
			for pixelIdx in range(self.feature_dim):
				pslice = batch[pixelIdx]
				# count the unique values
				pixelVal, pixelCount = np.unique(pslice, return_counts=True)
				# add it into likelihood
				for pV, pC in zip(pixelVal, pixelCount):
					self.likelihood[pixelIdx][pV][label] += pC 
			
			base += count

		# laplace smoothing
		k = 1
		self.likelihood += k
		
		# do division
		for class_idx in range(self.num_class):
			# likelihood = (# of times pixel i has value f in training examples from this class) / (Total # of training examples from this class)
			self.likelihood[:][:][class_idx] /= self.prior[class_idx]
			# prior = Total # of training example from this class / Total # of training example
			self.prior[class_idx] /= train_label.shape[0]

		# apply logrithm
		self.likelihood = np.log(self.likelihood)
		self.prior = np.log(self.prior)
		

	def test(self,test_set,test_label):
		""" Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
			by performing maximum a posteriori (MAP) classification.  
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

		accuracy = 0
		pred_label = np.zeros((len(test_set)))

		pass

		return accuracy, pred_label


	def save_model(self, prior, likelihood):
		""" Save the trained model parameters 
		"""    

		np.save(prior, self.prior)
		np.save(likelihood, self.likelihood)

	def load_model(self, prior, likelihood):
		""" Load the trained model parameters 
		""" 

		self.prior = np.load(prior)
		self.likelihood = np.load(likelihood)

	def intensity_feature_likelihoods(self, likelihood):
	    """
	    Get the feature likelihoods for high intensity pixels for each of the classes,
	        by sum the probabilities of the top 128 intensities at each pixel location,
	        sum k<-128:255 P(F_i = k | c).
	        This helps generate visualization of trained likelihood images. 
	    
	    Args:
	        likelihood(numpy.ndarray): likelihood (in log) with a dimension of
	            (# of features/pixels per image, # of possible values per pixel, # of class)
	    Returns:
	        feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
	            (# of features/pixels per image, # of class)
	    """
	    # YOUR CODE HERE
	    
	    feature_likelihoods = np.zeros((likelihood.shape[0],likelihood.shape[2]))

	    return feature_likelihoods