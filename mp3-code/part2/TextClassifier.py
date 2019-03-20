# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

from math import log, inf
from collections import Counter

"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """

        # count priors
        self.prior = { label:0 for label in set(train_label) }

        # likelihood dictonary, indexed by class->word->count
        self.likelihood = { label:{} for label in self.prior.keys() }


        # Count likelihoods
        for sentence, label in zip(train_set, train_label):
            self.prior[label] += 1
            for word in sentence:
                # increment count of a specific word in specific class
                if word in self.likelihood[label].keys():
                    (self.likelihood[label])[word] += 1
                else:
                    (self.likelihood[label])[word] = 2

        # add additional label for unseen word
        for label in self.prior:
            (self.likelihood[label])["_"] = 1

        for label in self.prior:

            # divide count for a specific word in a class by all words in class
            total_num_word_in_class = sum(self.likelihood[label].values())
            for word, count in self.likelihood[label].items():
                self.likelihood[label][word] = count / total_num_word_in_class
            
            # divide priors by total len of training set and apply log
            self.prior[label] = self.prior[label] / len(train_label)


    def predict(self, dev_set, dev_label,lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """

        accuracy = 0.0
        result = []
        
        for sentence in dev_set:
            
            Cval = { label:0 for label in self.prior }
            for label in self.prior:
                Cval[label] += log(self.prior[label])
                Cval[label] += sum(( log(self.likelihood[label][word]) for word in sentence if word in self.likelihood[label]  ))
                Cval[label] += sum(( log(self.likelihood[label]["_"]) for word in sentence if word not in self.likelihood[label] ))
            
            Cstar = max(Cval, key=lambda k: Cval[k])
            result.append(Cstar)


        accuracy = sum( 1 for i in range(len(result)) if result[i] == dev_label[i] ) / len(result)

        return accuracy,result

