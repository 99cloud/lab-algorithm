# import tensorflow as tf

import os
import numpy as np
# 防止在tf2下使用了tf1的API
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 只显示 warning 和 Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义计算图
learning_rate_ = tf.placeholder(dtype=tf.float32)
X_ = tf.placeholder(dtype=tf.float32, shape=[5])
y_ = tf.placeholder(dtype=tf.float32, shape=[5])
a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)

y_pred = a * X_ + b
loss = tf.constant(0.5) + tf.reduce_sum(tf.square(y_pred - y_))

# # 反向传播，手动计算变量（模型参数）的梯度
# grad_a = tf.reduce_sum((y_pred - y_) * X_)
# grad_b = tf.reduce_sum(y_pred - y_)
#
# # 梯度下降算法，手动跟新参数
# new_a = a - learning_rate_ * grad_a
# new_b = b - learning_rate_ * grad_b
# update_a = tf.assign(a, new_a)
# update_b = tf.assign(b, new_b)
#
# train_op = [update_a, update_b]

# 以上代码等价于
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_).minimize(loss)

# 数据流图定义到此结束
# 注意，直到目前，我们都没有进行任何实质的数据计算，仅仅是定义了一个数据图

num_epoch = 10000
learning_rate = 1e-3
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 7, 8, 10])

with tf.Session() as sess:
    # 初始化变量a和b
    tf.global_variables_initializer().run()
    # 循环将数据送入上面建立的数据流图中进行计算和更新变量
    for e in range(num_epoch):
        sess.run(train_op, feed_dict={X_: X, y_: y, learning_rate_: learning_rate})
    print(sess.run([a, b]))
