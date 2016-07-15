"""
    Coding by Sochi at 2016.07.13
"""
import numpy as np
import lrFunctions as func
from matplotlib import pyplot as plt


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

ini_theta = np.zeros((X.shape[1],1))

cost, grad = func.costFunction(ini_theta,X,Y,m)

print "Cost: ",cost,"  grad: ",grad