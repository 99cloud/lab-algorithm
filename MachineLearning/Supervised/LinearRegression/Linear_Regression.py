# -*- coding: utf-8 -*-
# encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Eg: y = 0.5 * x + 1
def load_dataset(n):
    noise = np.random.rand(n)
    x = [[0.1 * x] for x in range(n)]
    y = [(0.5 * x[i][0] + 1.0 + noise[i]) for i in range(n)]
    return np.array(x), np.array(y)

N = 100
x, y = load_dataset(N)
print('Data Shape', x.shape, y.shape)
# print(x, y)
# plt.scatter(x, y)
# plt.title('data')
# plt.show()

linreg = LinearRegression()
linreg.fit(x, y)
y_pred = linreg.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.title('Sklearn Linreg')

theta = np.hstack((linreg.intercept_, linreg.coef_))
print('Sklearn_Theta', theta)
loss = np.dot(y - y_pred, y - y_pred) / N
print('Sklearn_Linreg_Loss', loss)
plt.show()
# x0 = np.ones((x.shape[0], 1))
# X = np.hstack((x0, x))
# y_pred = X.dot(theta)