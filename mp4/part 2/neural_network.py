import numpy as np

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):

    #IMPLEMENT HERE
    Z1, cache = affine_forward(x_train, w1, b1)

    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):

    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn():
    pass

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    Z = np.matmul(np.c_[A, np.ones(A.shape[0])], np.r_[W, [b]])
    cache = A,W
    return Z, cache

def affine_backward(dZ, cache):
    A,W = cache
    dA = np.matmul(dZ,np.transpose(W))
    dW = np.matmul(np.transpose(A), dZ)
    dB = np.sum(dZ, axis=0)
    return dA, dW, dB

def relu_forward(Z):
    A = np.array(Z)
    A[A<0] = 0
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.where(Z > 0, dA, 0)
    return dZ

def cross_entropy(F, y):
    n, num_classes = F.shape

    # calculate Softmax
    exp_ij = np.exp(F)
    exp_i = np.sum(exp_ij, axis=1)

    # calculate loss
    F_iyi = F[np.arange(n), y.astype(int)]
    loss = np.sum(F_iyi - np.log(exp_i)) / n * -1
    
    # calculate dF
    one_hot = np.zeros(F.shape)
    one_hot[np.arange(n),y.astype(int)] = 1
    dF = (one_hot - (exp_ij / exp_i[:,None])) / n * -1

    return loss, dF
