import numpy as np
import numpy.linalg as lin

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunction(theta, X, y,m):

    grad = np.zeros((X.shape[1],1))
    h =  sigmoid(X.dot(theta))

    J  = (-np.log(h).T.dot(y)  - np.log(1-h).T.dot(1-y))/m
    grad = (X.T.dot(h-y))/m

    return J[0][0],grad