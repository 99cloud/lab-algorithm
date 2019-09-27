# K-Means

【[返回主仓](https://github.com/99cloud/lab-algorithm)】

## Catalog

- [说明](#说明)
- [K-Means介绍](#K-Means介绍)
	- [K-Means原理](#K-Means原理)
	- [K-Means步骤](#K-Means步骤)
	- [K-Means步骤](#K-Means步骤)
	- [K值的确定](#K值的确定)
	- [手肘法](#手肘法)
	- [K-Means与KNN](#K-Means与KNN)
- [K-Means优化](#K-Means优化)
	- [选取初始质心的位置](#选取初始质心的位置)
	- [处理距离计算效率低](#处理距离计算效率低)
		- [elkan_K-Means](#elkan_K-Means)
		- [大样本优化Mini_Batch_K-Means](#大样本优化Mini_Batch_K-Means)
- [sklearn的K-Means的使用](#sklearn的K-Means的使用)
	- [K-Means参数](#K-Means参数)
	- [K-Means使用](#K-Means使用)
- [小结](#小结)

# 说明

 ## 文档

此为非监督学习中，K-Means 的说明文档

**主要使用的包**

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import datasets
```

## 文件

| 文件           | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| KMeans_iris.py | KMeans 方法对 iris 数据集进行聚类的比较                      |
| kmeans.ipynb   | jupyter文档， Kmeans , Kmeans++ 及sklearn中官方KMeans 比较的示例<br />数据集包括 iris， boston， digits |
| kmeans_base.py | kmeans 基础算法                                              |
| kmeans_plus.py | kmeans++ 算法                                                |
| misc_utils.py  | 基础工具的函数库，包括距离计算，标签排序等                   |

# K-Means介绍

## 前言

机器学习按照有无标签可以分为 **监督学习** 和 **非监督学习** 

监督学习里面的代表算法就是： SVM 、逻辑回归、决策树、各种集成算法等等

非监督学习主要的任务就是通过一定的规则，把相似的数据聚集到一起，简称聚类

K-Means 算法是在非监督学习比较容易理解的一个算法，也是聚类算法中最著名的算法

## K-Means原理

K-Means 是典型的聚类算法，K-Means 算法中的 K 表示的是聚类为 K 个簇，Means 代表取每一个聚类中数据值的均值作为该簇的中心，或者称为质心，即用每一个的类的质心对该簇进行描述

## K-Means步骤

1. 创建 $k$ 个点作为起始质心
2. 计算每一个数据点到 $k$ 个质心的距离，把这个点归到距离最近的哪个质心
3. 根据每个质心所聚集的点，重新更新质心的位置
4. 重复2，3，直到前后两次质心的位置的变化小于一个阈值

整个变化的过程如果用图呈现出来会形象很多，下面的图就是 $k=2$ 的 K-Means 的过程

<img src='img\kmeans_step.png' width=500>

## K值的确定

K-Means 算法一般都只有一个超参数，就是 $k$ ，确定过程可以考虑如下步骤

1. 首先一个具体的问题肯定有它的具体的业务场景，$k$ 值需要根据业务场景来定义
2. 如果业务场景无法确定 $k$ 值，我们也有技术手段来找一个合适的 $k$ ，这个方法就是手肘法

## 手肘法

K-Means 算法中每一步都可以计算出 $loss$ 值又称为 $SSE$ 

$loss$ 值的计算方式就是每个聚类的点到它们质心的距离的平方
$$
SSE=\sum\limits_{i-1}^k\sum\limits_{x\in C_i}^{N_C} \vert x-\mu_i \vert ^2
$$
指定一个 $Max$ 值，即可能的最大类簇数，然后将类簇数 $k$ 从 $1$ 开始递增，一直到 $Max$ ，计算出 $Max$ 个 $SSE$ ，根据数据的潜在模式，当设定的类簇数不断逼近真实类簇数时，$SSE$ 呈现快速下降态势，而当设定类簇数超过真实类簇数时，$SSE$ 也会继续下降，但下降会迅速趋于缓慢，通过画出 $k-SSE$曲线，找出下降途中的拐点，即可较好的确定 $k$ 值

<img src='img\kmeans_k.png' width=400>

这样手肘图的拐点应该是 $k=4$ 的时候，所以我们可以定 $k=4$ 的时候聚类效果比较好

## K-Means与KNN

虽然 K-Means 和 KNN 名称接近，但两者其实差别还是很大的

- K-Means 是无监督学习的聚类算法，没有样本输出；而 KNN 是监督学习的分类算法，有对应的类别输出

- KNN 基本不需要训练，对测试集里面的点，只需要找到在训练集中最近的 $k$ 个点，用这最近的 $k$ 个点的类别来决定测试点的类别，而 K-Means 则有明显的训练过程，找到 $k$ 个类别的最佳质心，从而决定样本的簇类别

- 两者也有一些相似点，两个算法都包含一个过程，即找出和某一个点最近的点，两者都利用了最近邻 (nearest neighbors) 的思想

# K-Means优化

在使用 K-Means 过程中会遇到一些问题：如何选取初始质心的位置，如何处理距离计算的时候效率低的问题

## 选取初始质心的位置

假设已经确定了质心数 $k$ 的大小，那如何确定 $k$ 个质心的位置呢？事实上，$k$ 个初始化的质心的位置选择对最后的聚类结果和运行时间都有很大的影响，因此需要选择合适的 $k$ 个质心，如果仅仅是完全随机的选择，有可能导致算法收敛很慢，K-Means++ 算法就是对 K-Means 随机初始化质心的方法的优化

K-Means++ 的对于初始化质心的优化策略也很简单，如下：

1. 从输入的数据点集合中随机选择一个点作为第一个聚类中心 $\mu_1$

2. 对于数据集中的每一个点 $x_i$ ，计算它与已选择的聚类中心中最近聚类中心的距离
   $$
   D(x_i)=\arg\min \vert x_i−\mu_r \vert ^2,\qquad r=1,2,...k_{selected}
   $$

3. 选择一个新的数据点作为新的聚类中心，选择的原则是：$D(x)$ 较大的点，被选取作为聚类中心的概率较大

4. 重复 2. 和 3. 直到选择出 $k$ 个聚类质心

5. 利用这 $k$ 个质心来作为初始化质心去运行标准的 K-Means 算法

## 处理距离计算效率低

### elkan_K-Means

在传统的 K-Means 算法中，我们在每轮迭代时，要计算所有的样本点到所有的质心的距离，这样会比较的耗时，elkan K-Means 算法就是从这块入手加以改，它的目标是减少不必要的距离的计算

elkan K-Means 利用了两边之和大于等于第三边，以及两边之差小于第三边的三角形性质，来减少距离的计算

第一种规律是对于一个样本点 $x$ 和两个质心 $μ_{j1},μ_{j2}$ 如果我们预先计算出了这两个质心之间的距离 $D(j1,j2)$ ，则如果计算发现 $2D(x,j_1) \leq D(j_1,j_2)$ ，我们立即就可以知道 $D(x,j_1) \leq D(x,j_2)$ ，此时我们不需要再计算 $D(x,j_2)$ ，也就是说省了一步距离计算

第二种规律是对于一个样本点 $x$ 和两个质心 $\mu_{j1},\mu_{j2}$ ， 我们可以得到$D(x,j_2) \geq  \max\{0,D(x,j_1)-D(j_1,j_2)\}$ ，这个从三角形的性质也很容易得到。

利用上边的两个规律，elkan K-Means 比起传统的 K-Means 迭代速度有很大的提高，但是如果我们的样本的特征是稀疏的，有缺失值的话，这个方法就不适用了，此时某些距离无法计算，则不能使用该算法

### 大样本优化Mini_Batch_K-Means

在传统的 K-Means 算法中，要计算所有的样本点到所有的质心的距离，如果样本量非常大，比如达到 $10$ 万以上，特征有 $100$ 以上，此时用传统的 K-Means 算法非常的耗时，就算加上 elkan K-Means 优化也依旧缓慢，在大数据时代，这样的场景越来越多，此时 Mini Batch K-Means 应运而生

顾名思义，Mini Batch，也就是用样本集中的一部分的样本来做传统的 K-Means，这样可以避免样本量太大时的计算难题，算法收敛速度大大加快，当然此时的代价就是我们的聚类的精确度也会有一些降低，一般来说这个降低的幅度在可以接受的范围之内

在 Mini Batch K-Means 中，我们会选择一个合适的批样本大小 batch size ，我们仅仅用 batch size 个样本来做 K-Means 聚类，一般是通过无放回的随机采样得到 batch size 个样本

为了增加算法的准确性，我们一般会多跑几次 Mini Batch K-Means 算法，用得到不同的随机采样集来得到聚类簇，选择其中最优的聚类簇

# sklearn的K-Means的使用

## K-Means参数

- n_clusters : 聚类的个数k，default: 8
- init : 初始化的方式，default: [k-means++](##选取初始质心的位置)
- n_init : 运行 K-Means 的次数，最后取效果最好的一次， default: 10
- max_iter : 最大迭代次数， default: 300
- tol : 收敛的阈值，default: 1e-4
- n_jobs : 多线程运算，default = None，None代表一个线程，-1 代表启用计算机的全部线程
- algorithm : 有 'auto',  'full' or 'elkan' 三种选择，'full' 就是我们传统的 K-Means 算法，'elkan' 是我们讲的[elkan K-Means](###elkan K-Means) 算法，默认的 'auto' 则会根据数据值是否是稀疏的，来决定如何选择 'full' 和 'elkan' ，一般数据是稠密的，那么就是 'elkan' ，否则就是 'full' ，一般来说建议直接用默认的 'auto' 

## K-Means使用

```python
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_ 	# 输出原始数据的聚类后的标签值
>>> array([0, 0, 0, 1, 1, 1], dtype=int32)
kmeans.predict([[0, 0], [4, 4]]) 	# 根据已经建模好的数据，对新的数据进行预测
>>> array([0, 1], dtype=int32)
kmeans.cluster_centers_ 	# 输出两个质心的位置。
>>> array([[1., 2.],
       		 [4., 2.]])
```

KMeans 在 sklearn.cluster 的包里面，在 sklearn 里面都是使用 fit 函数进行聚类

顺便提一句，在 sklearn 中基本所有的模型的建模的函数都是 fit ，预测的函数都是 predict 

可以执行 Kmeans_iris.py 来进行鸢尾花数据分类的问题

<img src='img\kmeans_1.png' width=350><img src='img\kmeans_2.png' width=350>

<img src='img\kmeans_3.png' width=350><img src='img\kmeans_4.png' width=350>

1. 对数据用 $k=8$ 去聚类，因为数据本身只有 $3$ 类，所以聚类效果不好
2. 对数据用 $k=3$ 去聚类，效果不错
3. 还是用 $k=3$ 去聚类，但是改变初始化方式 init = random，n_init = 1，这样的随机初始化，并只运行 $1$ 次，最后的效果会不好
4. 最后一张图是数据本身的 label ，和右上相差不大

# 小结

K-Means的原理是很简单，但是我们仔细想想我们处理 K-Means 的思想和别的方法不太一样，先去猜想想要的结果，然后根据这个猜想去优化损失函数，再重新调整我们的猜想，一直重复这两个过程

其实这个猜想就是我们要求出的隐藏变量，优化损失函数的过程，就是最大化释然函数的过程，K-Means的算法就是一个 **EM 算法** 的过程



【[返回顶部](#K-Means)】

【[返回主仓](https://github.com/99cloud/lab-algorithm)】