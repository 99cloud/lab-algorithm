import numpy as np
from matplotlib import pyplot as plt


# Plot Function
def draw_pts(x, y):
    for i in range(len(x)):
        if y[i] == 1:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
#    plt.show()


def draw_line(w, b):
    line_x = [0, 7]
    line_y = [0, 0]
#   w * x +b
#   w[0] * x[0] + w[1] * x[1] + b = 0
    for i in range(len(line_x)):
        line_y[i] = (- w[0] * line_x[i] - b) / (w[1] + 1e-9)
    plt.plot(line_x, line_y)


# Data & Maker
# x = np.array([[3, 3], [4, 3], [2, 4], [1, 1], [1, 2], [2, 1]])
# y = np.array([1, 1, 1, -1, -1, -1])
num = 50
x = np.vstack((
    np.random.randn(num, 2) + 6, np.random.randn(num, 2) + 2
))
y = np.hstack((
    np.ones(num), - np.ones(num)
))

# draw_pts(x, y)

# Initial Parameter & Learning rate
w = [0, 0]
b = 0
lr = 1

# Primitive Form
for j in range(100):
    wrong_pt_cnt = 0
    for i in range(len(y)):
        if y[i] != np.sign(np.dot(w, x[i]) + b):
            w += lr * y[i] * x[i]
            b += lr * y[i]
            wrong_pt_cnt += 1
    if wrong_pt_cnt == 0:
        break

# draw_line(w, b)
# plt.show()

# Dual Form
gram = np.dot(x, x.T)
print('gram: \n', gram)

a = np.zeros(num * 2)
for j in range(100):
    wrong_pt_cnt = 0
    for i in range(len(y)):
        c = 0
        b = 0
        for k in range(len(y)):
            c += a[k] * y[k] * gram[k][i]
            b += a[k] * y[k]
        if y[i] != np.sign(c + b):
            a[i] += 1
            wrong_pt_cnt += 1
    if wrong_pt_cnt == 0:
        break
print('\na: \n', a)

w = [0, 0]
for k in range(len(y)):
    w += a[k] * y[k] * x[k]

plt.figure(figsize=(10, 5))
draw_pts(x, y)
draw_line(w, b)
plt.show()
