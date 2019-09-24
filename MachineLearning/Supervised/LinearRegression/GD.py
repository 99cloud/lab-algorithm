# -*- coding: utf-8 -*-
# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(n):
    noise = np.random.rand(n)
    x = [[0.1 * x] for x in range(n)]
    y = [(0.5 * x[i][0] + 1.0 + noise[i]) for i in range(n)]
    return np.array(x), np.array(y)

N = 100
x, y = load_dataset(N)
x0 = np.ones((x.shape[0], 1))
X = np.hstack((x0, x))


def sgd(X, y, iters, lr):
    theta = np.zeros(X.shape[1])
    cnt = 0
    while True:
        for i in range(X.shape[0]):
            theta += lr * (y[i] - np.dot(X[i], theta)) * X[i]
            cnt += 1
        if cnt >= iters:
            break
    return theta


def bgd(X, y, iters, lr, bs):
    theta = np.zeros(X.shape[1])
    cnt = 0
    while True:
        for i in range(int(np.ceil(X.shape[0] / bs))):
            begin = i * bs
            end = X.shape[0] if i * bs + bs > X.shape[0] else i * bs + bs
            X_batch = X[begin: end]
            y_batch = y[begin: end]
            theta += lr * np.dot(X_batch.T, y_batch - np.dot(X_batch, theta)) / X_batch.shape[0]
            cnt += 1
        if cnt >= iters:
            break
    return theta


theta = sgd(X, y, 5000, 0.01)
print('SGD_Theta', theta)
y_pred = X.dot(theta)
# print('SGD_y_pred\n', y_pred)
loss = np.dot(y - y_pred, y - y_pred) / N
print('SGD_Loss', loss)
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.title('SGD Linreg')


theta = bgd(X, y, 5000, 0.01, 10)
print('BGD_Theta', theta)
y_pred = X.dot(theta)
# print('BGD_y_pred\n', y_pred)
loss = np.dot(y - y_pred, y - y_pred) / N
print('BGD_Loss', loss)
plt.figure()
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.title('BGD Linreg')
plt.show()





