# 在tensorflow框架下编写python代码
import tensorflow

# （一）定义变量a，值为18（8分）
a = tensorflow.Variable(18, dtype=tensorflow.int32)
# （二）定义变量a乘以2的操作（8分）
a_2_mu = tensorflow.multiply(a, 2)
# （三）定义变量a除以3的操作（8分）
a_3_div = tensorflow.div(a, 3)
# （四）创建Session对象（8分）
with tensorflow.Session() as sess:
    # （五）执行全局变量初始化（8分）
    sess.run(tensorflow.global_variables_initializer())
    # （六）输出变量a的值（8分）
    print(sess.run(a))
    # （七）执行变量a乘以2的操作，输出结果（8分）
    print(sess.run(a_2_mu))
    # （八）把结果赋值给a（8分）
    a = tensorflow.assign(a, a_2_mu)
    # （九）执行变量a除以3的操作，输出结果（8分）
    print(sess.run(a_3_div))
    # （十）对张量[[2,3,4],[3,4,5]] 执行转置函数，参数用默认值，写出最后的结果（8分）
    print(sess.run(tensorflow.transpose([[2, 3, 4], [3, 4, 5]])))
    # （十一）完成100以内的整数的求和运算，输出每个步骤的结果。（每个步骤5分，至少4个步骤，共20分）
    sum1 = tensorflow.Variable(0, dtype=tensorflow.int32)
    sess.run(tensorflow.global_variables_initializer())
    for i in range(101):
        a = tensorflow.assign(sum1, tensorflow.add(sum1, i))
        print(sess.run(a))