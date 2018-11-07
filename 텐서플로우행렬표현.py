# import tensorflow as tf
#
# x = tf.constant([ [1.0,2.0,3.0] ])
# w = tf.constant([ [2.0],[2.0],[2.0] ])
# y = tf.matmul(x,w)
#
# print (x.get_shape())
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# result = sess.run(y)
#
# print(result)
#
#
# x = tf.Variable([ [1.,2.,3.] ], dtype=tf.float32)
# w = tf.constant([ [2.],[2.],[2.]], dtype=tf.float32)
# y = tf.matmul(x,w)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# result = sess.run(y)
#
# print(result)

import tensorflow as tf
input_data = [[1.,2.,3.],[1.,2.,3.],[2.,3.,4.]] #3x3
x1 = tf.placeholder(dtype=tf.float32, shape=[None,3])
w = tf.Variable([[2.],[2.],[2.]], dtype=tf.float32) #3x1
y = tf.matmul(x1,w)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

result = sess.run(y, feed_dict={x1:input_data})

print(result)