# Graph

基于networkx包，基本可以完成常规的图论任务

```shell
$ conda activate {env_name}
$ pip3 install networkx
# or
$ conda install networkx
```

**文件说明**

| 文件名                                                       | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| PyGraph.py                                                   | 代码，networkx的基本使用                                     |
| PyGraphDraw.py                                               | 代码，networkx的对Graph的呈现                                |
| PyPrim.py                                                    | 代码，Prim算法实现最小生成树                                 |
| PyGraphCPA.py                                                | 代码，获取关键路径算法                                       |
| PyMaxConSubgraph.py                                          | 代码，获取最大连通成分                                       |
| [MatrixOfGraph.ipynb](https://github.com/99cloud/lab-algorithm/tree/master/Normal/Graph/MatrixOfGraph.ipynb) | jupyter文档，与图论相关的矩阵，主要包括：邻接矩阵，关联矩阵，拉普拉斯矩阵，相关特征值，连通性等 |
| [GraphTheory.ipynb](https://github.com/99cloud/lab-algorithm/tree/master/Normal/Graph/GraphTheory.ipynb) | jupyter文档，networkx中内置算法的实例，主要包括，最短路径，最小/大生成树，拓扑排序，最大流，最小费用最大流，广度优先与深度优先 |
| networkx_reference.pdf                                       | pdf文档，networkx官方参考                                    |

