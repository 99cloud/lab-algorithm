from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
print("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

# 利用 X_train 中的数据创建 DataFrame
# 利用 iris_dataset.feature_names 中的字符串对数据进行标记
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 利用 DataFrame 创建散点图矩阵，按 y_train 着色
grr = pd.plotting.scatter_matrix(
    iris_dataframe, c=y_train, figsize=(12, 10), marker='o', hist_kwds={'bins': 20}, s=60, alpha=0.8, cmap=mglearn.cm3
)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95)
plt.show()
