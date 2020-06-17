import numpy as np
import matplotlib.pyplot as plt

import mglearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore", category=Warning)


def sklearn_normal_func(n=60):
    X, y = mglearn.datasets.make_wave(n_samples=n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    plt.figure(figsize=(10, 5))
    # Plot
    plt.title('Linear Regression')
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)

    # sklearn
    lr = LinearRegression().fit(X_train, y_train)
    plt.plot(X_train, lr.predict(X_train), 'r')

    # normal function
    X_T_X = np.linalg.inv(X_train.T.dot(X_train))
    theta = np.dot(X_T_X, X_train.T).dot(y_train)
    y_pred = X_train.dot(theta)
    plt.plot(X_train, y_pred, alpha=0.5)

    # legends and show
    plt.legend(["Sklearn Linear", "Normal Function Linear",
                "Train data", "Test data"], loc="best")
    plt.show()


if __name__ == '__main__':
    sklearn_normal_func()
