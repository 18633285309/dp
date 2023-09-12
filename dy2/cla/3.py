# 在tensorflow框架下编写python代码
# 1.定义变量a，值为[[1,3,5],[3,6,9]] （8分）
import tensorflow as tf

jiec = tf.Variable(1, dtype=tf.int32)
a = tf.Variable([[1, 3, 5], [3, 6, 9]])
# 2.定义变量b，值为[[2,7],[3,8],[2,6]] （8分）
b = tf.Variable([[2, 7], [3, 8], [2, 6]])
# 3.定义a数字乘以3的操作（8分）
a3 = tf.multiply(a, 3)
# 4.定义a矩阵乘以矩阵b的操作（8分）
a_b_m = tf.matmul(a, b)
# 5.创建Session对象（8分）
with tf.Session() as session:
    # 6.执行全局变量初始化（8分）
    session.run(tf.global_variables_initializer())
    # 7.输出变量a的值（8分）
    print(session.run(a))
    # 8.输出变量b的值（8分）
    print(session.run(b))
    # 9.执行a数字乘以3的操作，输出结果（8分）
    print(session.run(a3))
    # 10.执行a矩阵乘以矩阵b的操作，输出结果（8分）
    print(session.run(a_b_m))
    # 11.完成数字10以内的奇数的阶乘运算，输出每个步骤的结果。（每个步骤5分，至少4个步骤，共20分）
    for i in range(1, 10, 2):
        print(session.run(tf.assign(jiec, tf.multiply(jiec, i))))
