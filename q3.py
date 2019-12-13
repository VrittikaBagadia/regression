import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import os
import sys

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def compute_hessian(X,S):
	S = S * (1-S)
	examples, features = X.shape
	H = np.zeros((features, features))
	for i in range(features):
		for j in range(features):
			H[i][j] = 0
			for k in range(examples):
				H[i][j] += (-1) * X[k][i] * S[k] * X[k][j] 
	return(H)

def compute_gradient(y,x,S):
	temp = y - S
	theta0_grad = np.sum(temp * x[:,0])
	theta1_grad = np.sum(temp * x[:,1])
	theta2_grad = np.sum(temp * x[:,2])
	return(theta0_grad,theta1_grad,theta2_grad)

def log_likelihood(Y,S):
	ll = np.sum(Y * np.log(S) + (1-Y) * np.log(1 - S))
	return ll 

current_dir = os.getcwd()
dir_name = current_dir+'/plots/question3/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

xvalues = []
yvalues = []

# reading data
with open(sys.argv[1]) as inputX:
	X = csv.reader(inputX)
	xvalues = [[1,float(data[0]),float(data[1])] for data in X]
X = np.array(xvalues)

with open(sys.argv[2]) as inputY:
	Y = csv.reader(inputY)
	yvalues = [float(data[0]) for data in Y]
Y = np.array(yvalues)

# STEP 1: normalise x
for i in range(1,3,1):
	mean = np.mean(X[:,i])
	variance = np.std(X[:,i])
	X[:,i] = (X[:,i] - mean)/variance

sigmoid_v = np.vectorize(sigmoid)

theta_init = [0,0,0]
theta_init = np.array(theta_init)
theta_init = theta_init.transpose()
S = np.dot(X,theta_init)
S = sigmoid_v(S)

ll_old = log_likelihood(Y,S)
gradient = compute_gradient(Y,X,S)
H = compute_hessian(X,S)
H_inv = inv(H)

theta_new = theta_init - np.dot(H_inv, gradient)
S = np.dot(X,theta_new)
S = sigmoid_v(S)
ll_new = log_likelihood(Y,S)
change = ll_new - ll_old

while (abs(change) > 10**(-5)):
	
	# LL at old theta
	ll_old = ll_new
	theta = theta_new

	# update theta
	gradient = compute_gradient(Y,X,S)
	H = compute_hessian(X,S)
	H_inv = inv(H)
	theta_new = theta - np.dot(H_inv, gradient)

	# find LL at new theta
	S = np.dot(X,theta_new)
	S = sigmoid_v(S)
	ll_new = log_likelihood(Y,S)

	# difference in theta s
	change = ll_new - ll_old

## theta_new 
print('theta learnt: ')
print(theta_new)


## plotting
examples = X.shape[0]
zero = []
one = []
for i in range(examples):
	if (Y[i] == 0):
		zero.append([X[i][1], X[i][2]])
	else:
		one.append([X[i][1], X[i][2]])
zero = np.array(zero)
one = np.array(one)
plt.plot(zero[:,0], zero[:,1],'go', label = 'ZERO')
plt.plot(one[:,0], one[:,1], 'bo', label = 'ONE')

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
x_vals = np.arange(x_vals[0],x_vals[1],0.02)
y_vals = (-1) * (theta_new[0] + (theta_new[1] * x_vals))/theta_new[2]
plt.plot(x_vals, y_vals, 'r-')
plt.legend(loc = 'upper right')
plt.savefig(dir_name+'logistic regression.png')
plt.show()
