import tensorflow as tf
from MNISTLoader import *


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):         # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 100]
        x = self.dense2(x)          # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


def main():

    # define some hyper-parameters
    num_epochs = 5
    batch_size = 50
    learning_rate = 0.001

    # 实例化模型和数据读取类，并实例化一个 tf.keras.optimizer 的优化器
    model = MLP()
    data_loader = MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    '''迭代进行以下步骤
    1. 从 DataLoader 中随机取一批训练数据；
    2. 将这批数据送入模型，计算出模型的预测值；
    3. 将模型预测值与真实值进行比较，计算损失函数（loss）, 这里使用 tf.keras.losses 中的交叉熵函数作为损失函数；
    4. 计算损失函数关于模型变量的导数；
    5. 将求出的导数值传入优化器，使用优化器的 apply_gradients 方法更新模型参数以最小化损失函数
    '''
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    # 使用测试集评估模型的性能
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())


if __name__ == "__main__":
    main()
