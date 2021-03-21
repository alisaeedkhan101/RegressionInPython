#Polynomial Regression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/Ali Saeed/Downloads/insurance.csv')
df.head()

df = pd.concat([pd.Series(1, index=df.index, name='NF'), df], axis=1)
df.head()

X = df.drop(columns=['sex','age','children','smoker','region','charges'])
X['bmiSQRT'] = np.sqrt(X['bmi'])

y = df.iloc[:, 7] #dependent variable vector

global theta
theta = np.array([0]*len(X.columns))                    #generating 1x3 matrix for theta

m = len(df)

def hypothesis(theta, X):
    return theta * X

def computeCost(X, y, theta):
    y1 = hypothesis(theta, X)
    y1 = np.sum(y1, axis=1)
    return sum(np.sqrt((y1-y)**2))/(2*m)               #number of rows in dataset

def gradientDescent(X, y, theta, alpha, epochs):
    J = []                                              #cost function in each iterations
    k = 0
    while k < epochs:
        y1 = hypothesis(theta, X)
        y1 = np.sum(y1, axis=1)
        for c in range(0, len(X.columns)):
            theta[c] = theta[c] - alpha*(sum((y1-y)*X.iloc[:, c])/len(X))
        j = computeCost(X, y, theta) #updating J of theta
        J.append(j)
        k += 1
    return J, j, theta

def main():
    LR = 0.00001
    Itrs = 170
    global theta
    J, j, theta = gradientDescent(X, y, theta, LR, Itrs)

    y_hat = hypothesis(theta, X)
    y_hat = np.sum(y_hat, axis=1)

    plt.figure()
    plt.scatter(x=list(range(0, m)), y=y, color='blue')
    plt.scatter(x=list(range(0, m)), y=y_hat, color='red')
    plt.xlabel("Features")
    plt.ylabel("Cost")
    plt.title(f'Learning Rate {LR} and Iterations {Itrs}')
    plt.show()

    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("J of theta")
    plt.title('Local Minima in Gradient Descent')
    plt.scatter(x=list(range(0, len(J))), y=J)
    plt.show()

if __name__ == "__main__":
	main()