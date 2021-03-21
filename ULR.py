#Uni-Variable Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12.0, 9.0)
global X
global Y

def prep_data():
	data = pd.read_csv('C:/Users/Ali Saeed/Downloads/insurance.csv')
	global X
	X = data.iloc[:, 2]
	global Y
	Y = data.iloc[:, 6]

def calc_coeffs(alpha, epochs):
	#initialize both coefficients to zero before starting gradient descent
	theta1 = 0   #theta1 , m
	theta0 = 0   #theta0 , c
	#initializing the learning rate to a small value
	#alpha starting with 0.0001, 0.001
	#calculating the size of feature set i.e. number of rows in our datasset
	n = float(len(X))

	for i in range(epochs):
		Y_pred = theta1*X + theta0 					            #the current predicted value of Y
		D_theta1 = (1/n) * sum((Y_pred-Y)*X) 	                #Derivative wrt theta1
		D_theta0 = (1/n) * sum(Y_pred-Y) 		                #Derivative wrt theta0
		theta1 = theta1 - alpha * D_theta1; 					#update theta1
		theta0 = theta0 - alpha * D_theta0; 					#update theta0
	Y_pred = theta1*X + theta0

	plt.scatter(X, Y)
	plt.plot([min(X),max(X)],[min(Y_pred),max(Y_pred)],color='red')
	plt.xlabel("bmi")
	plt.ylabel("Cost")
	plt.title(f'Learning Rate {alpha} and Iterations {epochs}')
	plt.show()

def main():
	prep_data()
	calc_coeffs(0.001, 2000)                   #alpha and number of iterations

if __name__ == "__main__":
	main()