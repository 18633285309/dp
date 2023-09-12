import tensorflow as tf
xdata = [0,1,2]
ydata = [1.1,3.1,5.1]
X = tf.placeholder(dtype=tf.float32,shape=[None,])
Y = tf.placeholder(dtype=tf.float32,shape=[None,])
w = tf.Variable(tf.random_normal([]))
b = tf.Variable(tf.random_normal([]))
h = tf.multiply(X,w) + b
loss = tf.reduce_mean((h-Y)**2)
op = tf.train.GradientDescentOptimizer(0.02).minimize(loss)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(3001):
        loss_,op_,w_,b_ = session.run([loss,op,w,b],feed_dict={X:xdata,Y:ydata})
        if i % 100 == 0:
            print(i,loss_,w_,b_)