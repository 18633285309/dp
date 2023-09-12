# 使用Tensorflow框架实现单变量线性回归
# （一）导入tensorflow模块（8分）
import tensorflow as tf
# （二）设置随机种子（8分）
# 随机种子的目的是，让每次随机变量的结果都一样，好处是，让准确率和损失可复现
import tensorflow as tf
import numpy as np
# （三）初始化训练集， 4个点坐标值(x,y)分别为(0,1.1),(1,3.1),(2,5.1)。（8分）
xdata = [0,1,2]
ydata = [1.1,3.1,5.1]
# （四）设置预测模型函数（8分）
X = tf.placeholder(dtype=tf.float32,shape=[None,])
Y = tf.placeholder(dtype=tf.float32,shape=[None,])
w = tf.Variable(tf.random_normal([]))
b = tf.Variable(tf.random_normal([]))
h = tf.multiply(X,w) + b
# （五）设置代价或损失函数（8分）
loss = tf.reduce_mean((h-Y)**2)
# （六）创建梯度下降优化器，学习率设置为0.02（8分）
# （七）调用梯度下降优化器计算最小的代价（8分）
op = tf.train.GradientDescentOptimizer(0.02).minimize(loss)
# （八）创建Session（8分）
with tf.Session() as seesion:
    # （九）初始化全局变量（8分）
    seesion.run(tf.global_variables_initializer())
    # （十）进行3001次迭代（8分）
    for i in range(3001):
        # （十一）每100次输出一条信息（5分），包括当前迭代次数，代价值（5分）、W值（5分）、b值（5分）
        loss_,op_,w_,b_ = seesion.run([loss,op,w,b],feed_dict={X:xdata,Y:ydata})
        if i%100 == 0:
            print(i,loss_,w_,b_)
