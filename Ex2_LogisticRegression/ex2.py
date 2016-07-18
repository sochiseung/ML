"""
    Coding by Sochi at 2016.07.13
"""
import numpy as np
import lrFunctions as func
from matplotlib import pyplot as plt
import scipy.optimize as opt

# Part1. Data Import
f = open("ex2data1.txt", 'r')
lines = f.readlines()
xList = list()
yList = list()
for line in lines:
    values = line.replace('\n','').split(',')
    for i, v in enumerate(values):
        values[i] = float(v)
    xList.append(values[:-1])
    yList.append([int(values[-1])])
f.close()

X_2 = np.array(xList)
Y = np.array(yList)

m = len(Y)


#Part2. Compute Cost and Gradient

X_1 = np.ones((m,1))
X =  np.hstack((X_1,X_2))

theta = np.zeros((X.shape[1],1))


theta, cost, _, _, _ = \
        opt.fmin(func.costFunction, theta, full_output=True, maxiter= 400, args=(X,Y))

print 'Cost at theta found by fmin: %f' % cost
print 'theta: %s' % theta

#Part3. Predict

prob = func.sigmoid(np.array([[1,45,85]]).dot(theta.T))

print 'For a student with scores 45 and 85, we predict an admission ' \
      'probability of ', prob

p = func.predict(theta, X)
print 'Train Accuracy:', (p == Y).mean() * 100