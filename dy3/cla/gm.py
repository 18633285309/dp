import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]
print(np.array(x_data).shape)
print(np.array(y_data).shape)
# ②合理的运用tf.placeholder进行定义（7分）
X=tf.placeholder(dtype=tf.float32,shape=[None,3])
Y=tf.placeholder(dtype=tf.float32,shape=[None,1])
# ③合理的根据以上数据进行偏执和权重的设置，注意维度问题。（7分）
W=tf.Variable(tf.random_normal(shape=[3,1]))
b=tf.Variable(tf.random_normal(shape=[1]))
# ④完成预测模型（7分）
h=tf.matmul(X,W)+b
# ⑤用底层写出损失函数，注意是误差平方和。（7分）
loss=tf.reduce_mean((h-Y)**2)
# ⑥定义梯度下降（可以选择优化器种类也行）（7分）
op=tf.train.GradientDescentOptimizer(0.000000000001).minimize(loss)
# ⑦创建会话，初始化全局变量。（7分）
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    # ⑧进行迭代运算，要求1000次
    for i in range(1000):
        loss_,op_,W_,b_=sess.run([loss,op,W,b],feed_dict={X:x_data,Y:y_data})
    # ⑨合理的步数（可以是40步）给出损失值结果。（7分）
        if i%40==0:
            print(i,loss_)
    print(sess.run(h,feed_dict={X:x_data}))