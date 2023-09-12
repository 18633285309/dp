import tensorflow as tf
# 一、技能题（共100分)
# 在tensorflow框架下编写python代码
# （一）定义变量a，值为18（8分）
a = tf.Variable(18,dtype=tf.int32)
# （二）定义变量a乘以2的操作（8分）
a2 = tf.multiply(a,2)
# （三）定义变量a除以3的操作（8分）
a3 = tf.divide(a,3)
# （四）创建Session对象（8分）
with tf.Session() as session:
# （五）执行全局变量初始化（8分）
    session.run(tf.global_variables_initializer())
# （六）输出变量a的值（8分）
    print(session.run(a))
# （七）执行变量a乘以2的操作，输出结果（8分）
    res = session.run(a2)
    print(res)
# （八）把结果赋值给a（8分）
    session.run(tf.assign(a,res))
    print(session.run(a))
# （九）执行变量a除以3的操作，输出结果（8分）
    print(session.run(a3))
# （十）对张量[[2,3,4],[3,4,5]] 执行转置函数，参数用默认值，写出最后的结果（8分）
    tens = session.run(tf.constant([[2,3,4],[3,4,5]]))
    tT = tf.transpose(tens)
    print(session.run(tT))
# （十一）完成100以内的整数的求和运算，输出每个步骤的结果。（每个步骤5分，至少4个步骤，共20分）
    su = tf.Variable(0,dtype=tf.int32)
    session.run(tf.global_variables_initializer())
    for i in range(1,101):
        su = tf.add(su,i)
        print(session.run(su))

