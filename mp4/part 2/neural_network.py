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
 
    N = len(x_train)
    batch_size = 200
    losses = []

    shufflearg = np.arange(N)

    for ep in range(epoch):

        # print ("Epoch:", ep)

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

        # print ("Loss:", loss)

        if ep % 5 == 0:
            avg, _ = test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes)
            print ("Epoch:",ep,"Accuracy:", avg)

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

    classification = four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes, test=True)

    avg_class_rate = np.sum(np.equal(classification, y_test)) / len(x_test)

    class_rate_per_class = [0.0] * num_classes

    for c in range(num_classes):
        class_rate_per_class[c] = np.sum(classification[y_test==c]) / len(y_test==c)

    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, test=False):
    
    z1, acache1 = affine_forward(x_train, w1, b1)
    a1, rcache1 = relu_forward(z1)

    z2, acache2 = affine_forward(a1, w2, b2)
    a2, rcache2 = relu_forward(z2)

    z3, acache3 = affine_forward(a2, w3, b3)
    a3, rcache3 = relu_forward(z3)

    F, acache4 = affine_forward(a3, w4, b4)

    if test == True:
        classification = np.argmax(F, axis=1)
        return classification
    
    loss, dF = cross_entropy(F, y_train)

    dA3, dW4, dB4 = affine_backward(dF, acache4)

    dZ3 = relu_backward(dA3, rcache3)
    dA2, dW3, dB3 = affine_backward(dZ3, acache3)

    dZ2 = relu_backward(dA2, rcache2)
    dA1, dW2, dB2 = affine_backward(dZ2, acache2)

    dZ1 = relu_backward(dA1, rcache1)
    dX, dW1, dB1 = affine_backward(dZ1, acache1)

    eta = 0.1
    w1 -= eta * dW1
    w2 -= eta * dW2
    w3 -= eta * dW3
    w4 -= eta * dW4

    b1 -= eta * dB1
    b2 -= eta * dB2
    b3 -= eta * dB3
    b4 -= eta * dB4

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
    A = np.array(Z)
    A[A<0] = 0
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
