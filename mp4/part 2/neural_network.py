import numpy as np
import time

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
 
    start = time.time()
    N = len(x_train)
    batch_size = 200
    losses = []

    # default no shuffle, use arange
    shufflearg = np.arange(N)

    for ep in range(epoch):

        if shuffle:
            shufflearg = np.random.choice(N, size=N, replace=False)

        x_shuffle = x_train[shufflearg]
        y_shuffle = y_train[shufflearg]

        loss = 0

        for i in range(N//batch_size):
            
            x_batch = x_shuffle[i:i+batch_size]
            y_batch = y_shuffle[i:i+batch_size]

            loss += four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_batch, y_batch, num_classes, test=False)
        
        losses.append(loss)

        print ("Epoch:", ep, "Loss:", loss)

    end = time.time()
    print ("Elapsed:", end-start)

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

    # Get classification
    y_pred = four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes, test=True)

    # Get average rate by finding matchings
    avg_class_rate = np.sum(np.equal(y_pred, y_test)) / len(x_test)

    # Plot Confusion:
    # import plotting
    # plotting.plot_confusion_matrix(y_test, y_pred)

    # Get rate per class by finding matching indexes in y_test, then matching
    class_rate_per_class = [0.0] * num_classes
    for c in range(num_classes):
        argC = np.argwhere(y_test==c)
        class_rate_per_class[c] = np.sum(np.equal(y_pred[argC], c)) / len(argC)

    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, test=False):
    
    def forward(a,w,b):
        z, acache = affine_forward(a,w,b)
        a_n, rcache = relu_forward(z)
        return a_n, acache, rcache
    
    def backward(dA, acache, rcache):
        dZ = relu_backward(dA, rcache)
        dA_p, dW, dB = affine_backward(dZ, acache)
        return dA_p, dW, dB
    
    # forward prop
    a1, acache1, rcache1 = forward(x_train, w1, b1)
    a2, acache2, rcache2 = forward(a1, w2, b2)
    a3, acache3, rcache3 = forward(a2, w3, b3)
    # last layer no activation
    F, acache4 = affine_forward(a3, w4, b4)

    if test == True:
        classification = np.argmax(F, axis=1)
        return classification
    
    loss, dF = cross_entropy(F, y_train)

    # backprop
    dA3, dW4, dB4 = affine_backward(dF, acache4)
    dA2, dW3, dB3 = backward(dA3, acache3, rcache3)
    dA1, dW2, dB2 = backward(dA2, acache2, rcache2)
    dX, dW1, dB1 = backward(dA1, acache1, rcache1)

    # Update weights
    eta = 0.1
    for w, dW in zip((w1,w2,w3,w4),(dW1,dW2,dW3,dW4)):
        w -= eta * dW
    for b, dB in zip((b1,b2,b3,b4),(dB1,dB2,dB3,dB4)):
        b -= eta * dB

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
    cache = A, W
    return Z, cache

def affine_backward(dZ, cache):
    A, W = cache
    dA = np.matmul(dZ,np.transpose(W))
    dW = np.matmul(np.transpose(A), dZ)
    dB = np.sum(dZ, axis=0)
    return dA, dW, dB

def relu_forward(Z):
    cache = Z
    A = np.where(Z > 0, Z, 0)
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
