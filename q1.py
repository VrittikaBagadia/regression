import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
import os

def cost_function(y, x, theta0, theta1):
	res = (y - (theta1*x + theta0))**2
	J =  np.sum(res)
	J = J/(2)
	return J

def compute_gradient(y,x,theta0,theta1,learning_rate):
	derivative_theta0 = (y - (theta1*x + theta0)) * (-1) 
	derivative_theta1 = (y - (theta1*x + theta0)) * ((-1) * (x)) 
	theta0_grad = np.sum(derivative_theta0)
	theta1_grad = np.sum(derivative_theta1)
	theta0 = theta0 - learning_rate*theta0_grad
	theta1 = theta1 - learning_rate*theta1_grad
	return([theta0,theta1])

def run(y,x,theta0,theta1,learning_rate,convergence_factor):
	steps = []
	theta = []
	number_of_iterations = 0
	current_cost = cost_function(y,x,theta0,theta1)
	theta0_new, theta1_new = compute_gradient(y,x,theta0,theta1,learning_rate)
	new_cost = cost_function(y,x,theta0_new,theta1_new)
	number_of_iterations+=1
	diff = abs(current_cost-new_cost)
	steps.append([number_of_iterations,diff])			# steps stores - iteration number and corresponding decrease in cost function
	while (diff > convergence_factor and number_of_iterations < 100):
		theta0, theta1 = theta0_new, theta1_new
		current_cost = new_cost
		theta0_new, theta1_new = compute_gradient(y,x,theta0,theta1,learning_rate)
		theta.append([theta0_new,theta1_new])
		new_cost = cost_function(y,x,theta0_new,theta1_new)
		number_of_iterations+=1
		diff = abs(current_cost - new_cost)
		steps.append([number_of_iterations,diff])
	steps_array = np.array(steps)
	theta = np.array(theta)
	return(theta,theta0_new,theta1_new,new_cost,number_of_iterations,steps_array)

def gradient_descent(y,x,learning_rate,convergence_factor):
	theta0 = 0
	theta1 = 0
	theta, theta0_final, theta1_final, error, iterations_total, steps_array = run(y,x,theta0,theta1,learning_rate,convergence_factor)
	return[theta, theta0_final, theta1_final, error, iterations_total, steps_array]

current_dir = os.getcwd()
dir_name = current_dir+'/plots/question1/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

acidity = []
density = []

# reading data
with open(sys.argv[1]) as inputX:
	X = csv.reader(inputX)
	acidity = [float(data[0]) for data in X]
acidity = np.array(acidity)

with open(sys.argv[2]) as inputY:
	Y = csv.reader(inputY)
	density = [float(data[0]) for data in Y]
density = np.array(density)

# STEP 1: normalise x
mean = np.mean(acidity)
variance = np.std(acidity)
normalised_acidity = (acidity - mean)/variance

# STEP 2: set values of hyperparameters
learning_rate = float(sys.argv[3])
convergence_factor = 0.00001

theta, theta0_final, theta1_final, error, iterations_total, steps_array_0_003 = gradient_descent(density,normalised_acidity,learning_rate,convergence_factor)

# print('iterations: ' + str(steps_array_0_003.shape[0]))
print(" learning rate: " + str(learning_rate))
print(" theta1 or slope: " + str(theta0_final))
print("theta0 or intercept: " + str(theta1_final))


# PART B

plt.scatter(normalised_acidity,density)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
abline_values = [theta1_final*xi + theta0_final for xi in x_vals]
plt.plot(x_vals,abline_values,'r--')
plt.xlabel('acidity (normalised)')
plt.ylabel('density')
plt.savefig(dir_name+'hypothesis_function.png')
plt.show()


## PART C
# theta has thetas
# steps array has errors

time_gap = float(sys.argv[4])

x = np.arange(-0.75,2,0.01)
y = np.arange(-1,2,0.01)
xi, yi = np.meshgrid(x, y)
zi = np.zeros(xi.shape)
for i in range(xi.shape[0]):
	for j in range(xi.shape[1]):
		zi[i][j] = cost_function(density, normalised_acidity, xi[i][j], yi[i][j])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('J(theta)')
ax.plot_surface(xi, yi, zi)
for i in range(theta.shape[0]):
	plt.plot([theta[i][0]], [theta[i][1]], [steps_array_0_003[i][1]],'ro')
	plt.draw()
	plt.pause(0.001)
	time.sleep(time_gap)
plt.draw()
plt.savefig(dir_name+'mesh.png')
plt.show()


## PART D
x = np.arange(-2,4,0.01)
y = np.arange(-3,3,0.01)
xi, yi = np.meshgrid(x, y)
zi = np.zeros(xi.shape)
for i in range(xi.shape[0]):
	for j in range(xi.shape[1]):
		zi[i][j] = cost_function(density, normalised_acidity, xi[i][j], yi[i][j])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('J(theta)')
ax.contour(xi, yi, zi, 50)
for i in range(theta.shape[0]):
	plt.plot([theta[i][0]], [theta[i][1]], [steps_array_0_003[i][1]],'ro')
	plt.draw()
	plt.pause(0.001)
	time.sleep(time_gap)
plt.savefig(dir_name+'contours.png')
plt.show()








## choosing learning rate
# plt.plot(steps_array_0_001[:300,0],steps_array_0_001[:300,1],'ro-',label='0.001')
# plt.plot(steps_array_0_001[:300,0],steps_array_0_003[:300,1],'g^-',label='0.003')
# plt.plot(steps_array_0_001[:300,0],steps_array_0_01[:300,1],'bs-',label='0.01')
# plt.ylim(0,2)
# plt.legend(loc = 'upper right')
# plt.title("choosing learning rate")
# plt.xlabel("iteration number")
# plt.ylabel("error")
# plt.show()



