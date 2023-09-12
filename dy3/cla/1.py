# 使用Tensorflow框架实现单变量线性回归
# （一）导入tensorflow模块（8分）
import numpy as np
import tensorflow as tf
# （二）设置随机种子（8分）
tf.set_random_seed(888)
# （三）初始化训练集，x的值分别为0、1、2，y的值分别为2.2、4.2、6.2。（8分）
xdata = np.array([0,1,2])
ydata = [2.2,4.2,6.2]
# （四）设置预测模型函数（8分）
X = tf.placeholder(dtype=tf.float32,shape=[None,])
Y = tf.placeholder(dtype=tf.float32,shape=[None,])
w = tf.Variable(tf.random_normal([]))
b = tf.Variable(tf.random_normal([]))
h = tf.multiply(X,w) + b
# （五）设置代价或损失函数（8分）
loss = tf.reduce_mean((h-Y)**2)
# （六）创建梯度下降优化器，学习率设置为0.03（8分）
# （七）调用梯度下降优化器计算最小的代价（8分）
op = tf.train.GradientDescentOptimizer(0.03).minimize(loss)
# （八）创建Session（8分）
with tf.Session() as session:
# （九）初始化全局变量（8分）
    session.run(tf.global_variables_initializer())
# （十）进行2001次迭代（8分）
    for i in range(2001):
        loss_,op_,w_,b_ = session.run([loss,op,w,b],feed_dict={X:xdata,Y:ydata})
# （十一）每200次输出一条信息（5分），包括当前迭代次数，代价值（5分）、W值（5分）、b值（5分）
        if i % 200 ==0:
            print(i,loss_,w_,b_)

