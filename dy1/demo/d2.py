import tensorflow as tf
# 一、技能题（共100分)
# 在tensorflow框架下编写python代码
# （一）定义变量a，值为18（8分）
a = tf.Variable(18,dtype=tf.int32)
# （二）定义变量a乘以2的操作（8分）
b = tf.multiply(a,2)
# （三）定义变量a除以3的操作（8分）
c = tf.divide(a,3)
# （四）创建Session对象（8分）
with tf.Session() as session:
# （五）执行全局变量初始化（8分）
    session.run(tf.global_variables_initializer())
# （六）输出变量a的值（8分）
    print(session.run(a))
# （七）执行变量a乘以2的操作，输出结果（8分）
    res = session.run(b)
    print(res)
# （八）把结果赋值给a（8分）
    session.run(tf.assign(a,res))
# （九）执行变量a除以3的操作，输出结果（8分）
    session.run(c)
# （十）对张量[[2,3,4],[3,4,5]] 执行转置函数，参数用默认值，写出最后的结果（8分）
    t1 = tf.constant([[2,3,4],[3,4,5]])
    t1_op = tf.transpose(t1)
    print(session.run(t1_op))
# （十一）完成100以内的整数的求和运算，输出每个步骤的结果。（每个步骤5分，至少4个步骤，共20分）
    b = tf.Variable(0,dtype=tf.int32)
    for i in range(1, 101):
        partial_sum = session.run(tf.reduce_sum(tf.range(1, i+1)))
        print(f"步骤{i}的求和结果：{partial_sum}")
    print(session.run(tf.reduce_sum(tf.range(1,101))))
