# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the required Python libraries such as NumPy, Pandas, and Matplotlib.
2.Load the dataset and initialize the values for slope, intercept, learning rate, and number of iterations.
3.Apply the Gradient Descent formula to calculate the cost function and update the slope and intercept values repeatedly.
4.Train the linear regression model until the error is minimized and obtain the best fit line.
5.Display the regression line, predicted profit values, and print the final result of the model.
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: S Rithika
RegisterNumber: 212225040344

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("50_Startups.csv")
X_data = data.iloc[:,0].values.astype(float)  
y_data = data.iloc[:,4].values.astype(float)  
plt.figure(figsize=(6,4))
plt.scatter(X_data, y_data)
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")
def computeCost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    square_err = (h - y) ** 2
    return (1 / (2 * m)) * np.sum(square_err)
m = len(X_data)
X = np.append(np.ones((m,1)),
              X_data.reshape(m,1),
              axis=1)
y = y_data.reshape(m,1)
theta = np.zeros((2,1))
print(computeCost(X, y, theta))
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(), (predictions - y))
        descent = alpha * (1/m) * error
        theta = theta - descent
        J_history.append(computeCost(X, y, theta))
    return theta, J_history
theta, J_history = gradientDescent(
    X, y, theta, 0.0000000000001, 1500)
print("h(x) =" + str(round(theta[0,0],2)) +
      " + " + str(round(theta[1,0],6)) + "x1")
plt.figure(figsize=(6,4))
plt.plot(J_history[0:50])
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function using Gradient Descent")
plt.figure(figsize=(6,4))
plt.scatter(X_data, y_data)
x_value = np.array(X_data)
y_value = theta[0,0] + theta[1,0] * x_value
plt.plot(x_value, y_value, color='red')
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")
plt.show()
*/
```

## Output:

<img width="771" height="561" alt="ml3" src="https://github.com/user-attachments/assets/6180820c-feaa-4252-bf50-f30bb5a11907" />
<img width="785" height="517" alt="ml3(1)" src="https://github.com/user-attachments/assets/c0639afa-6e0b-465f-8e3c-4cf973573ca9" />
<img width="823" height="516" alt="image" src="https://github.com/user-attachments/assets/c3cec9b4-f999-4d13-a603-052a37352d69" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
