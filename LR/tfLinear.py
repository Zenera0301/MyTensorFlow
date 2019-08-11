# # -- encoding:utf-8 --
# """
# complicated by DingJing at 2019.8.8
# 用TensorFlow实现线性回归函数（输出更多参数）
# """
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# # 1. 构造数据
# np.random.seed(28)
# N = 100
# x_data = np.linspace(0, 6, N) + np.random.normal(loc=0.0, scale=2, size=N)
# y_data = 14 * x_data - 7 + np.random.normal(loc=0.0, scale=5.0, size=N)
# # 将x和y设置成为矩阵
# x_data.shape = -1, 1
# y_data.shape = -1, 1
#
# # 2. 定义两个placeholder存放输入数据
# x = tf.placeholder(tf.float32, [None, 1])
# y = tf.placeholder(tf.float32, [None, 1])
#
# # 3. 模型构建
# # 定义一个变量w和变量b
# w = tf.Variable(initial_value=tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0), name='w')
# b = tf.Variable(initial_value=tf.zeros([1]), name='b')
# # 构建一个预测值
# y_hat = w * x + b
# # 构建一个损失函数:以MSE作为损失函数（预测值和实际值之间的平方和）
# loss = tf.reduce_mean(tf.square(y_hat - y), name='loss')
# # 实现反向传播算法（以随机梯度下降的方式优化损失函数；在优化的过程中，是让loss函数最小化）
# train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
#
# # 4.运行
# with tf.Session() as sess:
#     # 初始化：全局变量更新
#     sess.run(tf.global_variables_initializer())
#     # 进行训练(n次)
#     for step in range(100):
#         # 模型训练
#         sess.run(train, feed_dict={x: x_data, y: y_data})
#         # 获得预测值
#         prediction_value = sess.run(y_hat, feed_dict={x: x_data})
#
#     # 画图显示拟合结果
#     plt.figure()
#     plt.scatter(x_data, y_data)  # 散点是真实值
#     plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
#     plt.show()






# -- encoding:utf-8 --
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1.构造数据
np.random.seed(28)
N = 100
x_data = np.linspace(0, 6, N) + np.random.normal(0.0,2.0,N)  # 当范围最大值为6时顺利，若为7 8 9可能有误，不知为何
y_data = 7 * x_data + 2 + np.random.normal(0.0,5,N)

# 画图显示拟合结果
# plt.figure()
# plt.scatter(x_data, y_data)
# plt.show()

# 调整矩阵形状为1列
x_data.shape = -1, 1
y_data.shape = -1, 1

# 2. 定义两个placeholder存放真实输入数据
x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="trueX")
y = tf.placeholder(dtype=tf.float32,shape=[None, 1], name="trueY")

# 3. 模型构建
# # 定义一个变量w和变量b
w = tf.Variable(initial_value=tf.zeros([1]),name='w')
# print('type:', type(w),'shape:', w.shape,'value:', w)
b = tf.Variable(initial_value=tf.zeros([1]),name='b')
# print('type:', type(b),'shape:', b.shape,'value:', b)
# 构建预测值
y_hat = w * x_data + b

# 4. 模型优化
# 构建一个损失函数:以MSE作为损失函数（预测值和实际值之间的平方和）
loss = tf.losses.mean_squared_error(y_hat,y)
# loss = tf.reduce_mean(tf.square(y_hat - y), name='loss')
# 实现反向传播算法（以随机梯度下降的方式优化损失函数；在优化的过程中，是让loss函数最小化）
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# train=tf.train.MomentumOptimizer(0.001,0.9).minimize(loss) #其他方法
# train=tf.train.AdamOptimizer(0.001).minimize(loss)


# 5. 运行
with tf.Session() as sess:
    # 初始化：全局变量更新
    sess.run(tf.global_variables_initializer())
    # 进行训练(n次)
    for step in range(100):
        # 模型训练
        sess.run(train, feed_dict={x: x_data, y: y_data})
        # 获得预测值
        prediction_value = sess.run(y_hat, feed_dict={x: x_data})
        # 每5轮打印一次w值
        if step % 5 == 0:
            print('After {} training steps,w is:{}\n'.format(step,w))
    # 打印最终权值
    print('Final w is:', sess.run(w))

    # 6.画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-', lw=5)
    plt.show()



