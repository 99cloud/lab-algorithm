import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from matplotlib import pyplot as plt


sample_num = 1000
sample_rate = 0.8   # 80% 的数据用于训练
data_split = int(sample_num * sample_rate)

x, y = make_classification(
    n_samples=sample_num,       # 生成样本的数量
    n_features=2,               # 生成样本的特征数， 等于后三者之和
    n_redundant=0,              # 多信息特征的个数
    n_informative=1,            # 冗余信息，informative 特征的随机线性组合
    n_clusters_per_class=1      # 某一个类别是由几个 cluster 构成的
)


# 训练数据和测试数据
x_data_train = x[:data_split, :]
x_data_test = x[data_split:, :]
y_data_train = y[:data_split]
y_data_test = y[data_split:]

# 正例和反例
positive_x1 = [x[i, 0] for i in range(sample_num) if y[i] == 1]
positive_x2 = [x[i, 1] for i in range(sample_num) if y[i] == 1]
negative_x1 = [x[i, 0] for i in range(sample_num) if y[i] == 0]
negative_x2 = [x[i, 1] for i in range(sample_num) if y[i] == 0]


# 定义感知机
clf = Perceptron(fit_intercept=True, n_iter_no_change=30, shuffle=True)
# 使用训练数据进行训练
clf.fit(x_data_train, y_data_train)

# 得到训练结果，权重矩阵
print('Coef Matrix:', clf.coef_)

# 决策函数中的常数，此处输出为：[0.]
print('Intercept:', clf.intercept_)

# 利用测试数据进行验证
acc = clf.score(x_data_test, y_data_test)
print('ACC:', acc)

# 画出正例和反例的散点图
plt.figure(figsize=(10, 5))
plt.scatter(positive_x1, positive_x2, c='red')
plt.scatter(negative_x1, negative_x2, c='blue')
# 画出超平面（在本例中即是一条直线）
# line_x = np.arange(-4, 4)
line_x = np.array([x.min(0)[0], x.max(0)[0]])
line_y = - (line_x * clf.coef_[0][0] + clf.intercept_) / clf.coef_[0][1]
plt.plot(line_x, line_y)
plt.ylim(x.min(0)[1] - 1, x.max(0)[1] + 1)
plt.show()
