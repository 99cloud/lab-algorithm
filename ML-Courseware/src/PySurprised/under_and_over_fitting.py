import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 将Pipeline封装 方便使用
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

x = np.random.uniform(-5, 5, size=100)  # 生成 x 特征 -5 到 5， 共 100 个
X = x.reshape(-1, 1)                    # 将 x 编程 100 行 1 列的矩阵
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)     # 模拟的是标记 y，对应的是 x 的二次函数

# 使用线性回归看下 score
reg = LinearRegression()
reg.fit(X, y)
reg.score(X, y)

# 画图
fig = plt.figure(figsize=(13, 4))


# 将预测值 y_pre 画图 对比真实 y
y_pre = reg.predict(X)
fig.add_subplot(1, 3, 1)
plt.scatter(x, y)
plt.plot(np.sort(x), y_pre[np.argsort(x)], color='r')
plt.title('Under-Fitting')
# plt.show()

# 查看MSE
y1_mse = mean_squared_error(y, y_pre)
print('y1_mse:', y1_mse)


def PolynomialRegression(degree):
    return Pipeline([                                   # 构建 Pipeline
        ("poly", PolynomialFeatures(degree=degree)),    # 构建 PolynomialFeatures
        ("std_scaler", StandardScaler()),               # 构建归一化 StandardScaler
        ("lin_reg", LinearRegression())                 # 构建线性回归 LinearRegression
    ])


# 设置 degree=2 进行 fit 拟合
poly2_reg = PolynomialRegression(2)
poly2_reg.fit(X, y)


# 将预测值 y2_pre 画图 对比真实 y
y2_pre = poly2_reg.predict(X)
fig.add_subplot(1, 3, 2)
plt.scatter(x, y)
plt.plot(np.sort(x), y2_pre[np.argsort(x)], color='r')
plt.title('Just-Right')
# plt.show()

# 求出MSE
y2_mse = mean_squared_error(y2_pre, y)
print('y2_mse:', y2_mse)

# 设置 degree=100 进行 fit 拟合
poly3_reg = PolynomialRegression(100)
poly3_reg.fit(X, y)


# 将预测值 y2_pre 画图 对比真实 y
y3_pre = poly3_reg.predict(X)
fig.add_subplot(1, 3, 3)
plt.scatter(x, y)
plt.plot(np.sort(x), y3_pre[np.argsort(x)], color='r')
plt.title('Over-Fitting')
plt.show()

# 求出MSE
y3_mse = mean_squared_error(y3_pre, y)
print('y3_mse:', y3_mse)
