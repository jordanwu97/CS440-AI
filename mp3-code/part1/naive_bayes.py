#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


class NaiveBayes(object):

    def __init__(
        self,
        num_class,
        feature_dim,
        num_value,
        ):
        """Initialize a naive bayes model. 

........This function will initialize prior and likelihood, where 
........prior is P(class) with a dimension of (# of class,)
............that estimates the empirical frequencies of different classes in the training set.
........likelihood is P(F_i = f | class) with a dimension of 
............(# of features/pixels per image, # of possible values per pixel, # of class),
............that computes the probability of every pixel location i being value f for every class label.  

........Args:
........    num_class(int): number of classes to classify 10
........    feature_dim(int): feature dimension for each example 784
........    num_value(int): number of possible values for each pixel 256
........"""

        self.num_value = num_value
        self.num_class = num_class
        self.feature_dim = feature_dim

        self.prior = np.zeros(num_class)
        self.likelihood = np.zeros((feature_dim, num_value, num_class))

    def train(self, train_set, train_label):
        """ Train naive bayes model (self.prior and self.likelihood) with training dataset. 
............self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,) 10
............self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of 
................(# of features/pixels per image, # of possible values per pixel, # of class). 784 x 256 x 10
............You should apply Laplace smoothing to compute the likelihood. 

........Args:
........    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim) 50000 x 784
........    train_label(numpy.ndarray): training labels with a dimension of (# of examples, ) 50000
........"""

        # Count priors, find args with label
        labelArgs = [np.nonzero(train_label == l)[0] for l in range(self.num_class)]
        self.prior = np.array([len(args) for args in labelArgs], dtype=float)

        # Rearrange train_set using sorted axis

        base = 0
        for (label, args) in zip(range(self.num_class), labelArgs):

            # cut a batch with same label out from arranged_train_set

            batch = np.transpose(train_set[args,:])

            # for each pixel in the batch...
            for pixelIdx in range(self.feature_dim):
                pslice = batch[pixelIdx]

                # count the unique values

                (pixelVal, pixelCount) = np.unique(pslice, return_counts=True)

                # add it into likelihood
                self.likelihood[pixelIdx,pixelVal,label] = pixelCount

            base += count

        # laplace smoothing
        k = 0.4
        self.likelihood += k

        # do divisions
        # likelihood = (# of times pixel i has value f in training examples from this class) / (Total # of training examples from this class)
        for classIdx in range(self.num_class):
            self.likelihood[:,:,classIdx] /= self.prior[classIdx]
            # prior = Total # of training example from this class / Total # of training example
            self.prior[classIdx] /= train_label.shape[0]

        # apply logrithm
        self.likelihood = np.log(self.likelihood)
        self.prior = np.log(self.prior)

    def test(self, test_set, test_label):
        """ Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
............by performing maximum a posteriori (MAP) classification.  
............The accuracy is computed as the average of correctness 
............by comparing between predicted label and true label. 

........Args:
........    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
........    test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

........Returns:
............accuracy(float): average accuracy value  
............pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
........"""

        # YOUR CODE HERE

        accuracy = 0
        pred_label = np.zeros(len(test_set))

        box = np.zeros((len(test_set), self.num_class, self.feature_dim
                       + 1))

        for classIdx in range(self.num_class):
            likelyhoodCurClass = self.likelihood[:, :, classIdx]
            subBox = box[:, classIdx, :]
            # retrieve likelyhood from test_sets pixel value
            subBox[:, np.arange(self.feature_dim)] = likelyhoodCurClass[np.arange(self.feature_dim), test_set[:, np.arange(self.feature_dim)]]
            subBox[:, self.feature_dim] = np.full(len(test_set), self.prior[classIdx])

        box = np.sum(box, axis=2)
        box = np.argmax(box, axis=1)

        accuracy = np.sum(np.equal(box, test_label)) / len(test_set)
        pred_label = box
        
        print ("Accuracy:", accuracy)

        return (accuracy, pred_label)

    def save_model(self, prior, likelihood):
        """ Save the trained model parameters 
........"""

        np.save(prior, self.prior)
        np.save(likelihood, self.likelihood)

    def load_model(self, prior, likelihood):
        """ Load the trained model parameters 
........"""

        self.prior = np.load(prior)
        self.likelihood = np.load(likelihood)

    def intensity_feature_likelihoods(self, likelihood):
        """
....    Get the feature likelihoods for high intensity pixels for each of the classes,
....        by sum the probabilities of the top 128 intensities at each pixel location,
....        sum k<-128:255 P(F_i = k | c).
....        This helps generate visualization of trained likelihood images. 
....    
....    Args:
....        likelihood(numpy.ndarray): likelihood (in log) with a dimension of
....            (# of features/pixels per image, # of possible values per pixel, # of class)
....    Returns:
....        feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
....            (# of features/pixels per image, # of class)
....    """

        feature_likelihoods = np.zeros((likelihood.shape[0],
                likelihood.shape[2]))

        sorted_likelihood = np.sort(likelihood, axis=1)
    
        feature_likelihoods = np.sum(sorted_likelihood[:,128:,:],axis=1)
        # print (feature_likelihoods[:,0])

        return feature_likelihoods
