# 朴素贝叶斯

【[返回主仓](https://github.com/99cloud/lab-algorithm)】

## Catalog

- [说明](#说明)
- [NaiveBayes介绍](#NaiveBayes介绍)
  - [NaiveBayes原理](#NaiveBayes原理)
  - [NaiveBayes算法](#NaiveBayes算法)
- [小结](#小结)

# 说明

 ## 文档

此为监督学习中，NaiveBayes 算法的说明文档

**主要使用的包**

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
```

## 文件

| 文件                  | 说明                                                         |
| --------------------- | ------------------------------------------------------------ |
| naiveBayes.ipynb      | jupyter文档，展示 朴素贝叶斯 和 高斯混合朴素贝叶斯 两种方法的简单实例 |
| naiveBayesBase.py     | 朴素贝叶斯算法实现                                           |
| naiveBayesGaussian.py | 高斯混合朴素贝叶斯算法实现                                   |
| word_utils.py         | 对于词组进行集合处理的函数                                   |

# NaiveBayes介绍

## 前言

说到朴素贝叶斯算法，首先涉及 **判别式** 和 **生成式** 的概念

- **判别式**：直接学习出特征输出 $Y$ 和特征 $X$ 之间的关系，如决策函数 $Y=f(X)$ ，或者从概率论的角度，求出条件分布 $P(Y|X)$ ，代表算法有 **决策树、KNN、逻辑回归、支持向量机、随机条件场CRF** 等
- **生成式**：直接找出特征输出 $Y$ 和特征 $X$ 的联合分布 $P(X,Y)$ ，然后用 $P(Y|X)=\frac{P(X,Y)}{P(X)}$ 得出，代表算法有 **朴素贝叶斯、隐式马尔可夫链** 等

## NaiveBayes原理

朴素贝叶斯算法基于 **贝叶斯定理和特征条件独立假设** 

- 贝叶斯定理：**先验概率+数据=后验概率** ，贝叶斯定理解决的是因 $X$ 无法直接观测、测量，而我们希望通过其结果 $Y$ 来反推出原因 $X$ 的问题，也就是知道一部分先验概率，来求后验概率的问题
	$$
	P(Y\vert X)=\frac{P(X\vert Y)P(Y)}{P(X)}=\frac{P(X|Y)P(Y)}{\sum_kP(X|Y=Y_k)P(Y_k)}
	$$

- 特征条件独立：特征条件独立假设 $Y$ 的 $n$ 个特征在类确定的条件下都是条件独立的，大大简化了计算过程，但是因为这个假设太过严格，所以会相应牺牲一定的准确率，这也是为什么称呼为 **朴素** 的原因

## NaiveBayes算法

**输入**：训练集为 $m$ 个样本 $n$ 个维度 $T=(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m)$ ，

​			共有 $K$ 个特征输出类别，分别为 $y\in\{c_1,c_2,\cdots,c_K\}$

**输出**：实例 $x_{(test)}$ 的分类

算法流程如下

1. 首先计算 $Y$ 的 $K$ 个 **先验概率**
	$$
	P(Y=c_k)
	$$

2. 然后计算条件概率分布
	$$
	P(X=x\vert Y=c_k)=P(X^{(1)}=x^{(1)},\cdots,X^{(n)}=x^{(n)}\vert Y=c_k)
	$$
	由于上式的参数是指数级别，无法计算，所以根据特征条件独立假设，可以化简为下式
	$$
	P(X=x|Y=c_k)=\prod\limits_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)	  \tag{1}
	$$

3. 根据贝叶斯原理，计算 **后验概率** 
	$$
	P(Y=c_k|X=x)=\frac {P(X=x|Y=c_k)P(Y=c_k)}{\sum_kP(X=x|Y=c_k)P(Y=c_k)}
	$$
	带入 $(1)$ 式，得到
	$$
	P(Y=c_k|X=x)=\frac{\prod\limits_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)P(Y=c_k)}
	{\sum_k\prod\limits_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)P(Y=c_k)}
	$$
	由于分母相同，上式再变为如下
	$$
	P(Y=c_k|X=x)=\prod\limits_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)P(Y=c_k)
	$$

4. 计算 $X(test)$ 的类别
	$$
	y_{(test)}=\arg \max\limits_{c_k} 
	\prod\limits_{j=1}^n P(X^{(j)}=x_{(test)}^{(j)}|Y=c_k)P(Y=c_k)
	$$

从上面的计算可以看出，没有复杂的求导和矩阵运算，因此效率很高

[**Gaussian Naive Bayes**](https://link.jianshu.com/?t=http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB) 属性特征值的条件概率
$$
P(x_i\vert y) = \frac1{\sqrt{2\pi\sigma_y^2}}\exp \left(
-\frac{(x_i-\mu_y)^2}{2\sigma_y^2} \right)
$$

# 小结

- 优点
	- 朴素贝叶斯模型发源于古典数学理论，**有稳定的分类效率**
	- **对小规模的数据表现很好** ，能个处理多分类任务，适合 **增量式训练** ，尤其是数据量超出内存时，我们可以一批批的去增量训练
	- **对缺失数据不太敏感** ，算法也比较简单，常用于文本分类
- 不足
	- 朴素贝叶斯模型的 **特征条件独立假设在实际应用中往往是不成立的**
	- 如果样本数据分布不能很好的代表样本空间分布，那先验概率容易测不准
	- 对输入数据的表达形式很敏感



【[返回顶部](#朴素贝叶斯)】

【[返回主仓](https://github.com/99cloud/lab-algorithm)】