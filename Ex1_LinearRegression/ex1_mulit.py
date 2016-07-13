"""
    Coding by Sochi at 2016.07.13
"""
import numpy as np
import functions1 as func


# Part1. Data Import
f = open("ex1data2.txt", 'r')
lines = f.readlines()
xList = list()
yList = list()
for line in lines:
    values = line.replace('\n','').split(',')
    for i, v in enumerate(values):
        values[i] = float(v)
    xList.append(values[:-1])
    yList.append([values[-1]])
f.close()

X_2 = np.array(xList)
Y = np.array(yList)

m = len(Y)


# Part2. Feature Normalize

normX, mu, sigma =func.featureNormalize(X_2)

X_1 = np.ones((m,1))
X = np.hstack((X_1,normX))

# Part3. Gradient Descent

# Choose some alpha value
alpha = 0.1;
num_iters = 50;

theta = np.zeros((X.shape[1],1))

theta,J_history = func.gradientDescent(X, Y, theta, alpha, num_iters ,m)

#print "Gradient Descent Theta: ", theta


# Predicted price of a 1650 sq-ft, 3 br house
t = np.array([[1650, 3]])
t = (t - mu) / sigma
predict = np.hstack((np.array([[1]]),t)).dot(theta)[0][0]

print 'Predicted price of a 1650 sq-ft, 3 br house (using GD):', predict

# Part6. Normal Equations

X = np.hstack((X_1,X_2))

theta = func.normalEquation(X,Y)

#print "Normal Equation Theta: ", theta

# Predicted price of a 1650 sq-ft, 3 br house
t = np.array([[1, 1650, 3]])
predict = t.dot(theta)[0][0]

print 'Predicted price of a 1650 sq-ft, 3 br house (using NE):', predict
