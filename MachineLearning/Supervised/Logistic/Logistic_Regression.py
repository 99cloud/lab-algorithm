# -*- coding: utf-8 -*-
# encoding=utf-8

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import random


def sigmoid(x):
    sgmd = (1 / (1 + np.exp(-x)))
    return sgmd


def logistic_regression(x, y, bs, lr, iters):
    x0 = np.ones((x.shape[0], 1))
    x = np.hstack((x0, x))
    theta = np.random.normal(0, 0.1, x.shape[1])

    start = 0
    for i in range(iters):
        end = start + bs
        if end > x.shape[0]:
            end = x.shape[0]
        x_batch = x[start:end]
        y_batch = y[start:end]

        start = end
        if start == x.shape[0]:
            start = 0

        y_pred = sigmoid(x_batch.dot(theta))
        grad = x_batch.T.dot(y_pred - y_batch)
        theta -= lr * grad
    return theta


def cal_loss(x, y, theta):
    x0 = np.ones((x.shape[0], 1))
    x = np.hstack((x0, x))
    p = sigmoid(x.dot(theta))
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    loss = np.mean(loss)
    return loss


def predict(x, theta):
    x0 = np.ones((x.shape[0], 1))
    x = np.hstack((x0, x))
    p = sigmoid(x.dot(theta))
    y_pred = (p > 0.5).astype('int')
    return y_pred


iris = load_iris()
# print(iris.DESCR)
x = iris.data
y = iris.target
print('x shape', x.shape)
print('y shape', y.shape)

idx = []
for i in range(y.shape[0]):
    if y[i] != 2:
        idx.append(i)

x_b = x[idx]
y_b = y[idx]
print('x_b shape', x_b.shape)
print('y_b shape', y_b.shape)

idx = list(range(x_b.shape[0]))
# print('idx', idx)
random.shuffle(idx)
# print('idx shuffle', idx)

ratio = 0.8
train_num = int(x_b.shape[0] * ratio)
train_idx = idx[:train_num]
test_idx = idx[train_num:]

x_train = x_b[train_idx]
y_train = y_b[train_idx]
x_test = x_b[test_idx]
y_test = y_b[test_idx]
print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)
print('x_test shape', x_test.shape)
print('y_test shape', y_test.shape)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)
print('w', classifier.coef_)
print('b', classifier.intercept_)
theta = np.hstack((classifier.intercept_.reshape(-1, 1), classifier.coef_))
print('theta', theta)
y_pred = classifier.predict(x_test)
print('y_pred', y_pred)
print('y_test', y_test)

theta = logistic_regression(x_train, y_train, 32, 0.01, 1000)
print('theta', theta)
loss = cal_loss(x_train, y_train, theta)
print('loss', loss)
y_pred = predict(x_test, theta)
print('y_pred', y_pred)
print('y_test', y_test)