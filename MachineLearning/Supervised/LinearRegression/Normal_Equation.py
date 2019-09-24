# -*- coding: utf-8 -*-
# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt

# Eg: y = 0.5 * x + 1
def load_dataset(n):
    noise = np.random.rand(n)
    x = [[0.1 * x] for x in range(n)]
    y = [(0.5 * x[i][0] + 1.0 + noise[i]) for i in range(n)]
    return np.array(x), np.array(y)

N = 100
x, y = load_dataset(N)
x0 = np.ones((x.shape[0], 1))
X = np.hstack((x0, x))

def normal_equation(X, y):
    # theta = inv(X'X)X'y
    X_T_X = np.linalg.inv(X.T.dot(X))
    theta = np.dot(X_T_X, X.T).dot(y)
    return theta

theta = normal_equation(X, y)
print('Normal_Equation_Theta', theta)
y_pred = X.dot(theta)
print('Normal_Equation_y_pred\n', y_pred)
loss = np.dot(y - y_pred, y - y_pred) / N
print('Normal_Equation_Loss', loss)
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.title('Normal Equation Linreg')
plt.show()

