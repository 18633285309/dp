import tensorflow as tf
# (1)按照要求去做线性回归运算；
# ①正确加载下图给予的亦或初始化数据（7分）
# x_data = [[73., 80., 75.],
#           [93., 88., 93.],
#           [89., 91., 90.],
#           [96., 98., 100.],
#           [73., 66., 70.]]
# y_data = [[152.],
#           [185.],
#           [180.],
#           [196.],
#           [142.]]
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
# ②合理的运用tf.placeholder进行定义（7分）
X = tf.placeholder(dtype=tf.float32,shape=[None,3])
Y = tf.placeholder(dtype=tf.float32,shape=[None,1])
# ③合理的根据以上数据进行偏执和权重的设置，注意维度问题。（7分）
w = tf.Variable(tf.random_normal(shape=[3,1]))
b = tf.Variable(tf.random_normal(shape=[1]))
# ④完成预测模型（7分）
h = tf.matmul(X,w) + b
# ⑤用底层写出损失函数，注意是误差平方和。（7分）
loss = tf.reduce_mean((h-Y)**2)
# ⑥定义梯度下降（可以选择优化器种类也行）（7分）
op = tf.train.AdamOptimizer(0.01).minimize(loss)
# ⑦创建会话，初始化全局变量。（7分）
with tf.Session()  as session:
    # ⑧进行迭代运算，要求1000次
    session.run(tf.global_variables_initializer())
    for i in range(1000):
    # ⑨合理的步数（可以是40步）给出损失值结果。（7分）
        loss_,op_,w_,b_ = session.run([loss,op,w,b],feed_dict={X:x_data,Y:y_data})
        if i % 40 == 0:
            print(i, loss_)
    # ⑩最后进入验证预测功能，要求加入正确的的注释（7分）
    print(session.run(h,feed_dict={X:x_data}))