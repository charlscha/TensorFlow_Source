import tensorflow as tf
# state = tf.Variable(0, name="counter")
#
# one         =tf.constant(1)
# new_value   =tf.add(state,one)
#
# update      =tf.assign(state,new_value)
#
# init_op        =tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(sess.run(state))
#     for a in range(3):
#         sess.run(update)
#         print(sess.run(state))
#
#         sess.close()

# node1 = tf.constant(3.0, dtype=tf.float32)
# node2 = tf.constant(4.0, dtype=tf.float32)
# node3 = tf.add(node1,node2)
#
# sess = tf.Session()
#
# print( sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
