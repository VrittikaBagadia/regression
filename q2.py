import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import sys
import os

def weight_matrix(x, X, bandwidth_parameter):
	examples_number = X.shape[0]
	W = np.zeros((examples_number, examples_number))
	for i in range(examples_number):
		W[i][i] = np.exp(((x - X[i][1])**2)/(-2 * bandwidth_parameter**2))
	return W

current_dir = os.getcwd()
dir_name = current_dir+'/plots/question2/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

xvalues = []
yvalues = []
with open(sys.argv[1]) as inputX:
	xvalues = csv.reader(inputX)
	X = [[1,float(data[0])] for data in xvalues]
X = np.array(X)

with open(sys.argv[2]) as inputY:
	yvalues = csv.reader(inputY)
	Y = [float(data[0]) for data in yvalues]
Y = np.array(Y)

# unweighted linear regression
theta_unweighted = np.dot(np.dot(inv(np.dot(X.transpose(),X)),X.transpose()), Y)
print(theta_unweighted)		# shape - 2 X 1

# plotting of hypothesis function
plt.figure(1)
# plt.subplot(121)
plt.title('UNWEIGHTED ')
# s = 'hypothesis function: Y = '+str(round(theta_unweighted[1],6))+'*X + '+str(round(theta_unweighted[0],6) )
abline_values = np.dot(theta_unweighted.transpose(), X.transpose())
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],abline_values,'r--')
# plt.text(-5,2.7,s, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(dir_name+'unweighted_hypothesis_fn.png')
plt.show()

## ---------------------------------------------------- ##

# weighted linear regression
minX = min(X[:,1])
maxX = max(X[:,1])
x_values = np.linspace(minX,maxX,50)		## includes end point by default
bandwidth_parameter = float(sys.argv[3])
theta_unweighted = []
y_learnt = []
for i in x_values:
	W = weight_matrix(i,X,bandwidth_parameter)
	theta = np.dot((np.dot((np.dot((inv(np.dot(np.dot(X.transpose(),W),X))),X.transpose())),W)),Y)
	theta_unweighted.append(theta)
	y_learnt.append( theta[1]*i + theta[0] )

# plt.subplot(122)
plt.title('WEIGHTED')
plt.scatter(X[:,1],Y)
plt.plot(x_values,y_learnt,'r-')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(dir_name+'locally_weighted_regression.png')
plt.show()

## ----------------------------------------------------- ##

# weighted linear regression with different bandwidth parameters
bandwidths = [0.1,0.3,2,10]
subplots = [221,222,223,224]
plt.figure(1)
plt.scatter(X[:,1],Y)
theta_unweighted = []
for i in range(4):
	y_learnt = []
	bandwidth = bandwidths[i]
	for j in x_values:
		W = weight_matrix(j,X,bandwidth)
		theta = np.dot((np.dot((np.dot((inv(np.dot(np.dot(X.transpose(),W),X))),X.transpose())),W)),Y)
		theta_unweighted.append(theta)
		y_learnt.append( theta[1]*j + theta[0] )
	plt.subplot(subplots[i])
	# axes = plt.gca()
	plt.title('bandwidth parameter: ' + str(bandwidth))
	plt.scatter(X[:,1],Y)
	plt.plot(x_values,y_learnt,'r-')
	# x_vals = np.array(axes.get_xlim())
	# y_vals = theta[0] + theta[1] * x_vals
	# plt.plot(x_vals, y_vals, 'r--')
	plt.xlabel('X')
	plt.ylabel('Y')
plt.savefig(dir_name+'different_tau.png')
plt.show()
