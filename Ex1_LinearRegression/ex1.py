"""

"""

# Part1. Plotting Data
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('ex1data1.csv')
# data.plot(kind='scatter',  x='population', y ='profit', color='Blue', label='group1')
# plt.title("Scatter plot of training data")
# plt.xlabel("Profit in $10,000s")
# plt.ylabel('Population of City in 10,000s')
# plt.show()


# Part2. Gradient Descent
print "GD"
Y = data[["profit"]].values
m = len(Y)
X_1 = np.ones((m,1))
X_2 = data[["population"]].values
X = np.hstack((X_1,X_2))
theta = np.zeros((2,1))

iterations = 1500;
alpha = 0.01;

def computeCost(X, y, theta, m):
    J = 0
    h = X.dot(theta)
    error = h - y
    A =  (((error.T).dot(error))[0][0])/(2*m)
    return A

def gradientDescent(X, y, theta, alpha, iterations,m):
    J_history = np.zeros((iterations, 1));

    for iter in range(iterations):
        h = X.dot(theta)
        error = h - y
        Z = (X.T.dot(error))/m
        theta = theta - alpha * Z
        J_history[iter] = computeCost(X, y, theta,m);
    return theta

print "initial Cost: ",computeCost(X, Y, theta,m)
theta = gradientDescent(X, Y, theta, alpha, iterations,m)

print "theta_0: ",theta[0],"\ttheta_1: ",theta[1]

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([[1, 3.5]]).dot(theta)[0][0] * 10000
print 'For population = 35,000, we predict a profit of ',predict1

predict2 = np.array([[1, 7]]).dot(theta)[0][0] * 10000
print 'For population = 70,000, we predict a profit of ',predict2
