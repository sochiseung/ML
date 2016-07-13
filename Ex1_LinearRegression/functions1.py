import numpy as np
import numpy.linalg as lin
def computeCost(X, y, theta, m):
    h = X.dot(theta)
    error = h - y
    J =  (((error.T).dot(error))[0][0])/(2*m)
    return J

def gradientDescent(X, y, theta, alpha, iterations,m):
    J_history = np.zeros((iterations, 1));

    for iter in range(iterations):
        h = X.dot(theta)
        error = h - y
        Z = (X.T.dot(error))/m
        theta = theta - alpha * Z
        J_history[iter] = computeCost(X, y, theta,m);
    return theta,J_history

def featureNormalize(X):
    X_norm = np.zeros(X.shape)
    mu = X.mean(0)
    sigma = X.std(0)
    #print "mu is ",mu,"  sigma is ",sigma
    for i,x in enumerate(X):
        X_norm[i] = (x-mu)/sigma
        #print "x is ",x, "\tnormalize is",(x - mu)/sigma
    return X_norm,mu,sigma

def normalEquation(X, y):
    theta = lin.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta