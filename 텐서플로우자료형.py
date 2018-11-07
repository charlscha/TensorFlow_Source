import tensorflow as tf

ph1 = tf.placeholder(dtype=tf.float32, shape=[3,3])
var1 = tf.Variable([1.,2.,3.],dtype=tf.float32, shape=[None,3])
const = tf.constant(1.,dtype=tf.float32)
image=[1,2,3]
label=[4,5,6]

feed_dict= {ph1 : image}



import tensorflow as tf

placeholder = tf.placeholder(tf.float32, shape=(3,3))
Variables = tf.Variable([1,2,3,4,5], dtype = tf.float32 )
constant = tf.constant([10,20,30,40,50], dtype = tf.float32)

sess = tf.Session()

result = sess.run(constant)
result

a = tf.constant([5], dtype=tf.float32)
b = tf.constant([6], dtype=tf.float32)
c = tf.constant([7], dtype= tf.float32)

d = a+b+c
sess.run(d)


var1 = tf.Variable([1], dtype=tf.float32)
var2 = tf.Variable([3], dtype=tf.float32)
var3 = tf.Variable([5], dtype=tf.float32)

var5 = var1 + var3 + var2
var5

init = tf.global_variables_initializer()

sess.run(init)
sess.run(var5)


value1 = 5
value2 = 3
value3 = 2

ph1 = tf.placeholder(dtype=tf.float32)
ph2 = tf.placeholder(dtype=tf.float32)
ph3 = tf.placeholder(dtype=tf.float32)

result_value = ph1 * ph2 * ph3
feed_dict = {ph1: value1, ph2: value2, ph3: value3 }

sess.run(result_value, feed_dict=feed_dict)
