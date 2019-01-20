#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 12:04:15 2019

@author: geo-lev
"""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_reg_train(t, X, lamda, winit, options):
    """inputs :
      t: N x 1 binary output data vector indicating the two classes
      X: N x (D+1) input data vector with ones already added in the first column
      lamda: the positive regularizarion parameter
      winit: D+1 dimensional vector of the initial values of the parameters
      options: options(1) is the maximum number of iterations
               options(2) is the tolerance
               options(3) is the learning rate eta
    outputs :
      w: the trained D+1 dimensional vector of the parameters"""

    w = winit

    # Maximum number of iteration of gradient ascend
    _iter = options[0]

    # Tolerance
    tol = options[1]

    # Learning rate
    eta = options[2]

    Ewold = -np.inf

    for i in range(_iter):
        yx = X.dot(w.T)
        s = sigmoid(yx)
        
        # Compute the cost function to check convergence
        Ew = np.sum(t*np.log(s)+(1-t)*np.log(1-s))-0.5*lamda * (w.T.dot(w))
        
        # Show the current cost function on screen
        print('iteration %d' % i)
        print('cost function :', Ew)

        # Break if you achieve the desired accuracy in the cost function
        if np.abs(Ew - Ewold) < tol:
            break

        # Gradient
        gradient = X.T.dot(t-s)-lamda*w
        
        # Update parameters based on gradient ascend
        w = w + eta*gradient

        Ewold = Ew

    return w

def log_reg_test(w, Xtest):
    # inputs :
    #   w: the D+1 dimensional vector of the parameters
    #   Xtest: Ntest x (D+1) input test data with ones already added in the first column
    # outputs :
    #   test: the predicted class labels
    #   ytest: Ntest x 1 vector of the sigmoid probabilities

    # Mean predictions
    ytest = sigmoid(Xtest.dot(w))

    # Hard classification decisions
    ttest = np.round(ytest)

    return ttest, ytest

def main():
    # train data file name
    train_file = 'data/data2Tr.txt'

    # read train txt file and store columns to X and t respectively
    train_data = np.loadtxt(train_file)
    X = train_data[:,:2]
    t = train_data[:,2]

    # test data file name
    test_file = 'data/data2Ts.txt'

    # read train txt file and store to Xtest
    test_data = np.loadtxt(test_file)
    Xtest = test_data

    # N of X
    N, D = X.shape

    # Ntest of Xtest
    Ntest = Xtest.shape[0]

    # add ones as first to X
    X = add_bias_2(X)

    # add ones as first to Xtest    
    Xtest = add_bias_2(Xtest)

    # initialize w for the gradient ascent
    # winit = np.zeros((1, D + 1))
    winit = np.zeros((D+1))

    # regularization parameter
    lamda = 0

    # options for gradient descent
    options = [500, 1e-6, 8 / N]

    # Train the model
    w = log_reg_train(t, X, lamda, winit, options)

    # test the model
    ttest, ytest = log_reg_test(w, Xtest)
    print (ttest.shape)

    x = X[:, 1]
    y = X[:, 2]
    plt.plot(x[t == 0], y[t == 0], 'r.', markersize = 10)
    plt.plot(x[t == 1], y[t == 1], 'b.', markersize = 10)
    

    minX = np.min(X[:, 2])
    maxX = np.max(X[:, 2])
    y = [- w[0] / w[2] - (w[1] / w[2]) * minX, - w[0] / w[2] - (w[1] / w[2]) * maxX]
    plt.plot([minX, maxX], y, 'k', linewidth = 2)
    plt.plot([minX, maxX], - np.log(0.9 / 0.1) / w[2] + y, 'k-.', linewidth = 1)
    plt.plot([minX, maxX], - np.log(0.1 / 0.9) / w[2] + y, 'k-.', linewidth = 1)
    plt.show()
    for i in range(Ntest):
        plt.plot(Xtest[i, 1], Xtest[i, 2], '.', markersize = 10, c = (1 - ytest[i], 0, ytest[i]))
    plt.show()

main()