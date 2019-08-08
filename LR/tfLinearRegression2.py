# -- encoding:utf-8 --
"""
complicated by DingJing at 2019.8.8
用TensorFlow实现线性回归函数（输出各个参数）
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. 构造数据
np.random.seed(28)
N = 100
x_data = np.linspace(0, 6, N) + np.random.normal(loc=0.0, scale=2, size=N)
y_data = 14 * x_data - 7 + np.random.normal(loc=0.0, scale=5.0, size=N)
# 将x和y设置成为矩阵
x_data.shape = -1, 1
y_data.shape = -1, 1

# 2. 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 3. 模型构建
# 定义一个变量w和变量b
# random_uniform：（random意思：随机产生数据， uniform：均匀分布的意思） ==> 意思：产生一个服从均匀分布的随机数列
# shape: 产生多少数据/产生的数据格式是什么； minval：均匀分布中的可能出现的最小值，maxval: 均匀分布中可能出现的最大值
w = tf.Variable(initial_value=tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0), name='w')
b = tf.Variable(initial_value=tf.zeros([1]), name='b')
# 构建一个预测值
y_hat = w * x + b

# 构建一个损失函数:以MSE作为损失函数（预测值和实际值之间的平方和）
loss = tf.reduce_mean(tf.square(y_hat - y), name='loss')

# # 以随机梯度下降的方式优化损失函数
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
# # 在优化的过程中，是让哪个函数最小化
# train = optimizer.minimize(loss, name='train')

# 一句话代替定义上面两句，实现反向传播算法（使用梯度下降算法训练）
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# 运行
# 打印函数
def print_info(r_w, r_b, r_loss):
    print("w={},b={},loss={}".format(r_w, r_b, r_loss))
with tf.Session() as sess:
    # 初始化：全局变量更新
    sess.run(tf.global_variables_initializer())
    # 输出初始化的w、b、loss
    r_w, r_b, r_loss = sess.run([w, b, loss], feed_dict={x: x_data, y: y_data})
    print_info(r_w, r_b, r_loss)
    # 进行训练(n次)
    for step in range(100):
        # 模型训练
        sess.run(train, feed_dict={x: x_data, y: y_data})
        # 输出训练后的w、b、loss
        r_w, r_b, r_loss = sess.run([w, b, loss], feed_dict={x: x_data, y: y_data})
        print_info(r_w, r_b, r_loss)
        prediction_value = sess.run(y_hat, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()