import tensorflow as tf
# 在tensorflow框架下编写python代码
# （一）定义一个常量c1，值为整数10（8分）
c1 = tf.constant(10,dtype=tf.int32)
# （二）定义一个变量a，值为整数100（8分）
a = tf.Variable(100,dtype=tf.int32)
# （三）定义另一个变量b，值为20（8分）
b = tf.Variable(20,dtype=tf.int32)
# （四）定义变量a与常量c1的和的操作（8分）
a_c1_sum = tf.add(a,c1)
# （五）定义两个变量a和b的和的操作（8分）
a_b_sum = tf.add(a,b)
# （六）定义两个变量a和b的差的操作（8分）
a_b_sub = tf.subtract(a,b)
# （七）创建Session对象（8分）
with tf.Session() as sesson:
    # （八）执行全局变量初始化（8分）
    sesson.run(tf.global_variables_initializer())
    # （九）输出变量a的值（6分）
    print(sesson.run(a))
    # （十）输出变量b的值（6分）
    print(sesson.run(b))
    # （十一）输出变量a与常量c1的和的值（8分）
    print(sesson.run(a_c1_sum))
    # （十二）输出两个变量a和b的和的值（8分）
    print(sesson.run(a_b_sum))
    # （十三）输出两个变量a和b的差的值（8分）
    print(sesson.run(a_b_sub))