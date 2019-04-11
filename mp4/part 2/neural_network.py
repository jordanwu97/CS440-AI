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
 
    losses = []

    W = np.array([w1,w2,w3,w4], dtype=object)
    B = np.array([b1,b2,b3,b4], dtype=object)

    for ep in range(epoch):

        randargs = np.random.choice(len(x_train), size=1000, replace=False)
        x_batch = x_train[randargs]
        y_batch = y_train[randargs]

        loss = four_nn(W,B,x_batch,y_batch, num_classes, test=False)
        losses.append(loss)

    print (losses)

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

    W = np.array([w1,w2,w3,w4], dtype=object)
    B = np.array([b1,b2,b3,b4], dtype=object)

    classification = four_nn(W,B,x_test,y_test, num_classes, test=True)

    print (classification)

    avg_class_rate = np.sum(np.equal(classification, y_test)) / len(x_test)
    class_rate_per_class = [0.0] * num_classes
    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(W, B, x, y, num_classes, test=False):

    def forward_prop(a,w,b):
        z, affineCache = affine_forward(a,w,b)
        a_1, activationCache = relu_forward(z)
        return a_1, affineCache, activationCache

    def back_prop(dA_1, activationCache, affineCache):
        dZ = relu_backward(dA_1, activationCache)
        dA, dW, dB = affine_backward(dZ, affineCache)
        return dA, dW, dB

    dW = np.array([ np.ones(w.shape) for w in W ], dtype=object)
    dB = np.array([ np.ones(b.shape) for b in B ], dtype=object)
    
    a1, affineCache1, activationCache1 = forward_prop(x, W[0], B[0])
    a2, affineCache2, activationCache2 = forward_prop(a1, W[1], B[1])
    a3, affineCache3, activationCache3 = forward_prop(a2, W[2], B[2])
    F, affineCache4 = affine_forward(a3,W[3],B[3])

    if test == True:
        classification = np.argmax(F, axis=1)
        return classification

    # back prop
    loss, dF = cross_entropy(F, y)
    dA4, dW[3], dB[3] = affine_backward(dF, affineCache4)
    dA3, dW[2], dB[2] = back_prop(dA4, activationCache3, affineCache3)
    dA2, dW[1], dB[1] = back_prop(dA3, activationCache2, affineCache2)
    dA1, dW[0], dB[0] = back_prop(dA2, activationCache1, affineCache1)

    # update weights
    eta = 0.1
    W -= eta * dW
    B -= eta * dB
    return loss

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
    softMax = (exp_ij / exp_i[:,None])

    # calculate loss
    F_iyi = F[np.arange(n), y.astype(int)]
    loss = np.sum(F_iyi - np.log(exp_i)) / n * -1
    
    # calculate dF
    one_hot = np.zeros(F.shape)
    one_hot[np.arange(n),y.astype(int)] = 1
    dF = (one_hot - softMax) / n * -1

    return loss, dF
