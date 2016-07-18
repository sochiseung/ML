import numpy as np
import numpy.linalg as lin

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunction(theta, X, y):
    m = X.shape[0]
    h =  sigmoid(X.dot(theta))

    J  = (-np.log(h).T.dot(y)  - np.log(1-h).T.dot(1-y))/m

    return J

def gradientDescent(theta, X, y):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    error = h - y
    grad = (X.T.dot(error))/m
    return grad

def predict(theta, X):
    p = sigmoid(X.dot(theta)) >= 0.5
    return p.reshape(X.shape[0],1)