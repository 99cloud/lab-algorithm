# TensorFlow 部分部分模型测试

## 开始前建议完善一些包

```bash
pip install -r requirement.txt
```

## 测试

- 使用 MLP 模型对 MNIST 数据集的测试

    ```bash
    python MLP.py
    ```

- 使用 CNN 模型对 MNIST 数据集的测试

    ```bash
    python CNN.py
    ```

- 如果是 GPU 版本，还可以体验一下  CNN 模型的测试，在 CPU 版本上运行较慢

    ```bash
    python MobileNetV2.py
    ```

- 查看一个基于 TensorFlow 的分类 Demo

    ```bash
    jupyter-notebook
    ```

    