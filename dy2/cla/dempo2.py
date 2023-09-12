# 一、技能题（共100分)
# 在tensorflow框架下编写python代码
import tensorflow as tf
# 1.定义变量a，值为24（8分）
a = tf.Variable(24,dtype=tf.int32)
summ = tf.Variable(0,dtype=tf.int32)
# 2.定义变量a乘以3的操作（8分）
a3 = tf.multiply(a,3)
# 3.定义变量a除以2的操作（8分）
ac2 = tf.divide(a,2)
# 4.创建Session对象（8分）
with tf.Session() as session:
    # 5.执行全局变量初始化（8分）
    session.run(tf.global_variables_initializer())
    # 6.输出变量a的值（8分）
    print(session.run(a))
    # 7.执行变量a乘以3的操作，输出结果（8分）
    print(session.run(a3))
    # 8.把结果赋值给a（8分）
    print(session.run(tf.assign(a,a3)))
    # 9.执行变量a除以2的操作，输出结果（8分）
    print(session.run(ac2))
    # 10.对张量[[7,2,8],[8,1,9]] 执行转置函数，参数用默认值，写出最后的结果（8分）
    print(session.run(tf.transpose([[7,2,8],[8,1,9]])))
    # 11.完成100以内所有的偶数的求和运算，输出每个步骤的结果。（每个步骤5分，至少4个步骤，共20分）
    for i in range(0,101,2):
        print(session.run(tf.assign(summ,tf.add(summ,i))))