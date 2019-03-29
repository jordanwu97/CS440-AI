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
        self.lambda_mixture = 0.5

    def fit_bigram(self, train_set, train_label):
        self.bigram_likelihood = { label:{ w_1:{} for w_1 in self.likelihood[label] } for label in self.likelihood }

        for sentence, label in zip(train_set, train_label):
            self.prior[label] += 1
            for w_idx in range(1,len(sentence)):
                # bigram model 
                w_1 = sentence[w_idx - 1]
                w_2 = sentence[w_idx - 1]

                if w_2 not in self.bigram_likelihood[label][w_1]:
                    self.bigram_likelihood[label][w_1][w_2] = 1

                self.bigram_likelihood[label][w_1][w_2] += 1
        
        t_all = 0

        for label in self.bigram_likelihood:
            for w_1 in self.bigram_likelihood[label]:
                
                # add additional label for unseen w_2
                self.bigram_likelihood[label][w_1]["_"] = 0.1

                t = sum(self.bigram_likelihood[label][w_1].values())
                t_all += t

                for w_2 in self.bigram_likelihood[label][w_1]:
                    self.bigram_likelihood[label][w_1][w_2] = self.bigram_likelihood[label][w_1][w_2] / t

        for label in self.bigram_likelihood:
            self.bigram_likelihood[label]["_"]["_"] = min( self.bigram_likelihood[label][w_1]["_"] for w_1 in self.bigram_likelihood[label])
                
        print (self.bigram_likelihood[1]["abbott"])

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
        self.likelihood = { label:{} for label in self.prior }


        # Count likelihoods
        for sentence, label in zip(train_set, train_label):
            self.prior[label] += 1
            for word in sentence:
                # increment count of a specific word in specific class
                if word not in self.likelihood[label]:
                    self.likelihood[label][word] = 1
                self.likelihood[label][word] += 1

        for label in self.prior:

            # add additional label for unseen word            
            (self.likelihood[label])["_"] = 0.1

            # divide count for a specific word in a class by all words in class
            total_num_word_in_class = sum(self.likelihood[label].values())
            for word, count in self.likelihood[label].items():
                self.likelihood[label][word] = count / total_num_word_in_class
            
            # divide priors by total len of training set and apply log
            self.prior[label] = self.prior[label] / len(train_label)

        self.fit_bigram(train_set, train_label)

    def predict_bigram(self, dev_sentence):

        P = { label:0 for label in self.prior }

        for label in self.prior:
            # multiply prior and P(w_1)
            P[label] += log(self.prior[label])
            
            dev_sentence.insert(0,"=")
            bigram_sentence = [(dev_sentence[i-1],dev_sentence[i]) for i in range(1, len(dev_sentence))]
            
            P[label] += sum(log(self.bigram_likelihood[label][w_1][w_2]) for (w_1, w_2) in bigram_sentence if (w_1 in self.bigram_likelihood[label] and w_2 in self.bigram_likelihood[label][w_1] ))
            P[label] += sum(log(self.bigram_likelihood[label][w_1]["_"]) for (w_1, w_2) in bigram_sentence if (w_1 in self.bigram_likelihood[label] and w_2 not in self.bigram_likelihood[label][w_1] ))
            P[label] += sum(log(self.likelihood[label][w_2]) for (w_1, w_2) in bigram_sentence if (w_1 not in self.bigram_likelihood[label] and w_2 in self.likelihood[label]))
            P[label] += sum(log(self.bigram_likelihood[label]["_"]["_"]) for (w_1, w_2) in bigram_sentence if (w_1 not in self.bigram_likelihood[label] and w_2 not in self.likelihood[label] ))

        return P

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
            
            P = { label:0 for label in self.prior }

            for label in self.prior:
                P[label] += log(self.prior[label])
                P[label] += sum(( log(self.likelihood[label][word]) for word in sentence if word in self.likelihood[label]  ))
                P[label] += sum(( log(self.likelihood[label]["_"]) for word in sentence if word not in self.likelihood[label] ))
            
            P_bigram = self.predict_bigram(sentence)

            self.lambda_mixture = 0.01

            for label in P:
                mixed = (1-self.lambda_mixture)*P[label] + (self.lambda_mixture)*P_bigram[label]
                P = {label:(1-self.lambda_mixture)*P[label] + (self.lambda_mixture)*P_bigram[label] for label in P}

            Cstar = max(P, key=lambda k: P[k])
            result.append(Cstar)


        accuracy = sum( 1 for i in range(len(result)) if result[i] == dev_label[i] ) / len(result)

        return accuracy,result

