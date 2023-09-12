import tensorflow as tf

# (一) 定义变量a，值为18
a = tf.Variable(18, dtype=tf.float32)
# (二) 定义变量a乘以2的操作
multiply_op = tf.multiply(a, 2)
# (三) 定义变量a除以3的操作
divide_op = tf.divide(a, 3)
# (四) 创建Session对象
sess = tf.Session()
# (五) 执行全局变量初始化
sess.run(tf.global_variables_initializer())
# (六) 输出变量a的值
print("变量a的值：", sess.run(a))
# (七) 执行变量a乘以2的操作，输出结果
result = sess.run(multiply_op)
print("变量a乘以2的结果：", result)
# (八) 把结果赋值给a
sess.run(tf.assign(a, result))
# (九) 执行变量a除以3的操作，输出结果
result = sess.run(divide_op)
print("变量a除以3的结果：", result)
# (十) 对张量[[2,3,4],[3,4,5]] 执行转置函数，参数用默认值，写出最后的结果
input_tensor = tf.constant([[2,3,4],[3,4,5]])
transpose_op = tf.transpose(input_tensor)
result = sess.run(transpose_op)
print("转置后的结果：", result)
# (十一) 完成100以内的整数的求和运算，输出每个步骤的结果
total_sum = tf.reduce_sum(tf.range(1, 101))
for i in range(1, 101):
    partial_sum = sess.run(tf.reduce_sum(tf.range(1, i+1)))
    print("当前步骤{}的求和结果：{}".format(i, partial_sum))
print("最终求和结果：", sess.run(total_sum))

# 关闭Session
sess.close()
