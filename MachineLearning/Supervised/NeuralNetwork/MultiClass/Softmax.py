import numpy as np
# import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import random

def one_hot(y, class_num):
    one_hot_code = np.zeros((y.shape[0], class_num))
    for i in range(one_hot_code.shape[0]):
        one_hot_code[i][y[i]] = 1
    return one_hot_code

def softmax(x):
    sum_x = np.sum(np.exp(x), axis=1)
    softmax_out = x.copy()
    for i in range(x.shape[0]):
        softmax_out[i] = np.exp(x[i]) / sum_x[i]
    return softmax_out

def sgd_softmax(x, y, class_num, bs, lr, iters):
    x0 = np.ones((x.shape[0], 1))
    x = np.hstack((x0, x))
    theta = np.random.normal(0, 0.1, (class_num, x.shape[1]))

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
        y_pred = softmax(x_batch.dot(theta.T))
        y_one_hot = one_hot(y_batch, class_num)
        grad = x_batch.T.dot(y_pred - y_one_hot)
        grad = grad.T
        theta = theta * (1 - 1e-5) - lr * grad / (x_batch.shape[0])
    return theta

def cal_loss(x, y, theta):
    x0 = np.ones((x.shape[0], 1))
    x = np.hstack((x0, x))
    p = softmax(x.dot(theta.T))
    y = one_hot(y, 3)
    loss = -(y * np.log(p))
    loss = np.mean(loss)
    return loss

def predict(x, theta):
    x0 = np.ones((x.shape[0], 1))
    x = np.hstack((x0, x))
    p = softmax(x.dot(theta.T))
    y_pred = np.argmax(p, axis=1)
    return y_pred

iris = load_iris()
# print(iris.DESCR)
x = iris.data
y = iris.target
print('x shape', x.shape)
print('y shape', y.shape)

idx = list(range(x.shape[0]))
# print('idx', idx)
random.shuffle(idx)
# print('idx shuffle', idx)

ratio = 0.8
train_num = int(x.shape[0] * ratio)
train_idx = idx[:train_num]
test_idx = idx[train_num:]

x_train = x[train_idx]
y_train = y[train_idx]
x_test = x[test_idx]
y_test = y[test_idx]
print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)
print('x_test shape', x_test.shape)
print('y_test shape', y_test.shape)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)
theta = np.hstack((classifier.intercept_.reshape(-1, 1), classifier.coef_))
print('theta', theta)
loss = cal_loss(x_train, y_train, theta)
print('loss', loss)
y_pred = classifier.predict(x_test)
print('y_pred', y_pred)
y_pred = predict(x_test, theta)
print('y_pred', y_pred)
print('y_test', y_test)

theta = sgd_softmax(x_train, y_train, 3, 32, 0.01, 1000)
print('theta',theta)
loss = cal_loss(x_train, y_train, theta)
print('loss', loss)
y_pred = predict(x_test, theta)
print('y_pred', y_pred)
print('y_test', y_test)