from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


# 定义目标函数
def f(x):
    return x ** 2 + 10 * np.sin(x)


def f_plot():
    # 绘制目标函数的图形
    plt.figure(figsize=(10, 5))
    x = np.arange(-10, 10, 0.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('optimize')
    plt.plot(x, f(x), 'r-', label='$f(x)=x^2+10sin(x)$')
    # 图像中的最低点函数值
    a = f(-1.3)
    plt.annotate('min', xy=(-1.3, a), xytext=(3, 40), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.legend()
    plt.show()


def f_xmin():
    optimize.fmin_bfgs(f, 0)

    grid = (-10, 10, 0.1)
    xmin_global = optimize.brute(f, (grid,))
    print(xmin_global)

    x0 = -10
    xmin_global_2 = optimize.basinhopping(f, x0, stepsize=5).x
    print(xmin_global_2)


def main():
    f_plot()
    f_xmin()


if __name__ == '__main__':
    main()
