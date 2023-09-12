import tensorflow as tf
import numpy as np
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]
X = tf.placeholder(dtype=tf.float32,shape=[None,2])
Y = tf.placeholder(dtype=tf.float32,shape=[None,1])
w = tf.Variable(tf.random_normal([2,1]))
b = tf.Variable(tf.random_normal([1]))
h = tf.sigmoid(tf.matmul(X,w)+b)
y_pred = tf.cast(h>0.5,tf.int32)
loss = -tf.reduce_mean(Y*tf.log(h)+(1-Y)*tf.log(1-h))
op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(1000):
        loss_,op_ = session.run([loss,op],feed_dict={X:x_data,Y:y_data})
        if i % 40 ==0:
            print(i,loss_)
    print(session.run(y_pred,feed_dict={X:[x_data[-1]]}))