# Mglearn

## Code of Local Running
1. 切换到 Python3 环境，可以用虚拟环境 virtualenv，总之要确认 `python --version` 的输出是 `Python3` 字样

	```console
	$ python --version
	Python 3.7.6
	$ pip --version
	pip 20.0.2 from /Users/FDUHYJ/anaconda3/envs/option/lib/python3.7/site-packages/pip (python 3.7)
	```

1. 切换回 `Mglearn/` 目录，安装 pip 依赖

	```console
	$ ls
	README.md                             mglearn_for_supervised_learning       requirements.txt                      tree.dot
    cache                                 mglearn_for_supervised_learning.ipynb tmp
    images                                mytree.dot                            tmp.png

	$ pip install -r requirements.txt
	```

1. 使用 `jupyter notebook` 查看 `mglearn_for_supervised_learning.ipynb` 文档，最后 Ctrl + C 退出 

    ```console
    $ jupyter notebook
    ...

    关闭服务 (y/[n])y
    ```

1. 或者直接查看 [Markdown 文档](https://github.com/99cloud/lab-algorithm/tree/master/MachineLearning/Mglearn/mglearn_for_supervised_learning/mglearn_for_supervised_learning.md)
