# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

from math import log,exp
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


        self.count_bigram_w1w2 = { label:{ w_1:{} for w_1 in self.unigram_count_w[label] } for label in self.unigram_count_w }

        # count(w_2,w_1)
        for sentence, label in zip(train_set, train_label):
            for w_idx in range(1,len(sentence)):
                # bigram model 
                w_1 = sentence[w_idx - 1]
                w_2 = sentence[w_idx]

                self.count_bigram_w1w2[label][(w_1,w_2)] = self.count_bigram_w1w2[label].get((w_1,w_2), 0) + 1
        
    def predict_bigram(self, sentence):

        P_bigram = { label:0 for label in self.prior }

        for label in self.prior:
            # multiply prior and P(w_1)
            P_bigram[label] += log(self.prior[label])

            P_bigram[label] += log((self.unigram_count_w[label].get(sentence[0], 0) + 10) / (sum(self.unigram_count_w[label].values()) + len(self.V)))
            
            bigram_sentence = [(sentence[i-1],sentence[i]) for i in range(1, len(sentence))]

            def p(w1,w2):
                return log((self.count_bigram_w1w2[label].get((w1,w2),0) + 1) / (self.unigram_count_w[label].get(w1,0) + len(self.V)))

            P_bigram[label] += sum(p(w1,w2) for (w1,w2) in bigram_sentence) 


        return P_bigram
                

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

        # count dictonary, indexed by class->word->count
        self.unigram_count_w = { label:{} for label in self.prior }


        # Count w's
        for sentence, label in zip(train_set, train_label):
            self.prior[label] += 1
            for word in sentence:
                # increment count of a specific word in specific class
                self.unigram_count_w[label][word] = self.unigram_count_w[label].get(word,0) + 1

        # for label in self.prior:

        #     # divide count for a specific word in a class by all words in class
        #     total_num_word_in_class = sum(self.likelihood[label].values())
        #     for word in self.likelihood[label]:
        #         self.likelihood[label][word] = self.likelihood[label][word] / total_num_word_in_class
            
        #     # divide priors by total len of training set and apply log
        #     self.prior[label] = self.prior[label] / len(train_label)

        # Vocabulary set
        self.V = set()
        for label in self.prior:
            self.V |= set(self.unigram_count_w[label].keys())

        self.fit_bigram(train_set, train_label)

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

        result = []

        V = len(self.V)
        
        for i, sentence in enumerate(dev_set):
            
            P_unigram = { label:0 for label in self.prior }

            for label in self.prior:
                P_unigram[label] += log(self.prior[label])

                N = sum(self.unigram_count_w[label].values())

                def p(w):
                    return log((self.unigram_count_w[label].get(w, 0) + 10) / (N + V))

                P_unigram[label] += sum(p(w) for w in sentence)

            P_bigram = self.predict_bigram(sentence)
            
            P_mix = { label:0 for label in self.prior }

            def argMax(A):
                return max(A, key=lambda k: A[k])

            self.lambda_mixture = 0.5

            # apply mixture
            P_mix = {label:(1-self.lambda_mixture)*exp(P_unigram[label]) + (self.lambda_mixture)*exp(P_bigram[label]) for label in P_mix}

            # append Cstar
            result.append(argMax(P_mix))


        accuracy = sum( 1 for i in range(len(result)) if result[i] == dev_label[i] ) / len(result)

        return accuracy,result

