import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv,det
import math
import os
import sys

current_dir = os.getcwd()
dir_name = current_dir+'/plots/question4/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
x = [i.strip().split() for i in open(sys.argv[1]).readlines()]
x = np.array(x)
X = x.astype(float)

y = [0 if i.strip()=='Alaska' else 1 for i in open(sys.argv[2]).readlines()]
Y = np.array(y).astype(float)

mu_0 = np.array([0.0,0.0])
mu_0[0] = np.sum((1-Y) * X[:,0])/np.sum(1-Y)
mu_0[1] = np.sum((1-Y) * X[:,1])/np.sum(1-Y)

mu_1 = np.array([0.0,0.0])
mu_1[0] = np.sum(Y * X[:,0])/np.sum(Y)
mu_1[1] = np.sum(Y * X[:,1])/np.sum(Y)

examples = Y.shape[0]
MU = []
for i in range(examples):
	if (y[i] == 0):
		MU.append([mu_0[0],mu_0[1]])
	else:
		MU.append([mu_1[0],mu_1[1]])
MU = np.array(MU)

a1 = X.transpose() - MU.transpose()
a2 = X - MU
covariance_matrix = np.dot(a1,a2)/examples

## PART A
print("SAME COVARIANCE MATRIX")
print("mu0")
print(mu_0)
print("mu1")
print(mu_1)
print("covariance network")
print(covariance_matrix)


## PART B (plotting)
zero = []
one = []
for i in range(examples):
	if (y[i] == 0):
		zero.append(X[i])
	else:
		one.append(X[i])
zero = np.array(zero)
one = np.array(one)
plt.plot(zero[:,0], zero[:,1],'go', label = 'Alaska')
plt.plot(one[:,0], one[:,1], 'bo', label = 'Canada')
plt.legend(loc = 'upper right')
plt.savefig(dir_name+'data_plotted.png')
plt.show()

## PART C
phi = float(one.shape[0]) / examples
rhs = 2 * np.log(phi/(1-phi)) - ( np.dot(np.dot(mu_0.transpose(),inv(covariance_matrix)),mu_0) ) + ( np.dot(np.dot(mu_1.transpose(),inv(covariance_matrix)),mu_1 ))
rhs = rhs / 2
lhs = np.dot( (mu_1.transpose() - mu_0.transpose()) , inv(covariance_matrix) )
c1 = lhs[0]
c2 = lhs[1]
plt.figure(1)
plt.plot(zero[:,0], zero[:,1],'go', label = 'Alaska')
plt.plot(one[:,0], one[:,1], 'bo', label = 'Canada')
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
x_vals = np.arange(x_vals[0],x_vals[1],0.02)
y_vals = [(rhs - c1*xi)/c2 for xi in x_vals]
y_vals = np.array(y_vals)
plt.plot(x_vals,y_vals,'y-')
plt.legend(loc = 'upper right')
plt.savefig(dir_name + 'linear_decision_boundary.png')
plt.show()

## PART D
zeroes = []
ones = []
for i in range(examples):
	if (y[i] == 0):
		zeroes.append(x[i])
	else:
		ones.append(x[i])
zeroes = np.array(zeroes)
ones = np.array(ones)
temp_mat_0 = zeroes.astype(float) - mu_0.astype(float)
temp_mat_1 = ones.astype(float) - mu_1.astype(float)
covariance_matrix_0 = np.dot(temp_mat_0.transpose(),temp_mat_0) / zeroes.shape[0]
covariance_matrix_1 = np.dot(temp_mat_1.transpose(),temp_mat_1) / ones.shape[0]

print("DIFFERENT COVARIANCE MATRICES")
print("mu0")
print(mu_0)
print("mu1")
print(mu_1)
print("covariance network 0")
print(covariance_matrix_0)
print("covariance matrix 1")
print(covariance_matrix_1)

## PART E
M1 = inv(covariance_matrix_0) - inv(covariance_matrix_1)
M2 = np.dot(mu_0.transpose(),inv(covariance_matrix_0)) - np.dot(mu_1.transpose(),inv(covariance_matrix_1))
M2 = -2 * M2
constant = np.dot(np.dot(mu_0.transpose(),inv(covariance_matrix_0)),mu_0) - np.dot(np.dot(mu_1.transpose(),inv(covariance_matrix_1)),mu_1) + 2*math.log(phi/(1-phi)) + math.log(abs(det(covariance_matrix_0)) / abs(det(covariance_matrix_1)))
coeff_x0_2 = M1[0][0]
coeff_x1_2 = M1[1][1]
coeff_x0_x1 = M1[0][1] + M1[1][0]
coeff_x0 = M2[0]
coeff_x1 = M2[1]
ansset1 = []
ansset2 = []
for xii in x_vals:
	a = coeff_x1_2
	b = coeff_x1 + (coeff_x0_x1 * xii)
	c = (coeff_x0_2 * xii * xii) + (coeff_x0 * xii) + constant
	D = (b*b - 4*a*c)**0.5
	ans1 = ((-1)*b + D)/(2*a)
	ans2 = ((-1)*b - D)/(2*a)
	ansset1.append(ans1)
	ansset2.append(ans2)
plt.plot(zero[:,0], zero[:,1],'go', label = 'Alaska')
plt.plot(one[:,0], one[:,1], 'bo', label = 'Canada')
plt.plot(x_vals,y_vals,'y-')
plt.plot(x_vals,ansset1,'r-')
plt.plot(x_vals,ansset2,'r-')
plt.legend(loc = 'upper right')
plt.savefig(dir_name+'quadratic_boundary.png')
plt.show()

# lhs =np.log(det(covariance_matrix_1)/det(covariance_matrix_0)) + 2*np.log((1 - phi)/phi) +  np.dot((np.dot(mu_0.transpose(),inv(covariance_matrix_0))),mu_0) + np.dot((np.dot(mu_1.transpose(),inv(covariance_matrix_0))),mu_1)
# rhs2 = 2*(np.dot(mu_1.transpose(),inv(covariance_matrix_1)) - np.dot(mu_0.transpose(),inv(covariance_matrix_0)))
# c1 = rhs2[0]
# c2 = rhs2[1]

# coeff_x0_2 = covariance_matrix_0[0][0] - covariance_matrix_1[0][0]
# coeff_x1_2 =  covariance_matrix_0[1][1] - covariance_matrix_1[1][1]
# coeff_x0_x1 = covariance_matrix_0[0][1] + covariance_matrix_0[1][0] - covariance_matrix_1[0][1] - covariance_matrix_1[1][0]
# coeff_x0 = c1
# coeff_x1 = c2
# coeff_1 = (-1) * lhs

# ## x_vals there
# ansset1 = []
# ansset2 = []
# for xii in x_vals:
# 	a = coeff_x1_2
# 	b = coeff_x1 + (xii * coeff_x0_x1)
# 	c = coeff_1 + (xii * coeff_x0) + (coeff_x0_2 * xii * xii)

# 	D = ((b*b) - (4*a*c))**0.5
# 	ans1 = ((-1)*b + D)/(2*a)
# 	ans2 = ((-1)*b - D)/(2*a)
# 	ansset1.append(ans1)
# 	ansset2.append(ans2)
# ansset1 = np.array(ansset1)
# ansset2 = np.array(ansset2)
# plt.plot(x_vals,ansset1,'r-')
# plt.plot(x_vals,ansset2,'r-')
# plt.show()


## ---------------------------------------------







