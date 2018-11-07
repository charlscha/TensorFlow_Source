import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

input_data =[[1,5,4,7,8,10,12],
             [5,8,10,3,9,7,1]]
label_data =[[0,0,0,1,0],
             [1,0,0,0,0]]

INPUT_SIZE = 7
HIDDEN_SIZE_1 = 10
HIDDEN_SIZE_2 = 8
CLASSES = 5


x = tf.placeholder(tf.float32, shape = [None, INPUT_SIZE] )
y = tf.placeholder(tf.float32, shape = [None, CLASSES] )

tensor_map = {x : input_data, y : label_data}


weight_hidden_1 = tf.Variable( tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN_SIZE_1]) , dtype=tf.float32 )
bias_hidden_1 = tf.Variable( tf.zeros(shape=[HIDDEN_SIZE_1]) , dtype=tf.float32 )
hidden_1 = tf.sigmoid(tf.matmul( x , weight_hidden_1 ))  + bias_hidden_1

weight_hidden_2 = tf.Variable( tf.truncated_normal(shape=[HIDDEN_SIZE_1, HIDDEN_SIZE_2]) , dtype=tf.float32 )
bias_hidden_2 = tf.Variable( tf.zeros(shape=[HIDDEN_SIZE_2]) , dtype=tf.float32 )
hidden_2 = tf.sigmoid(tf.matmul( hidden_1 , weight_hidden_2 ) ) + bias_hidden_2

weight_output = tf.Variable( tf.truncated_normal(shape=[HIDDEN_SIZE_2, CLASSES]) , dtype=tf.float32 )
bias_output = tf.Variable( tf.zeros(shape=[CLASSES]) , dtype=tf.float32 )
answer = tf.sigmoid(tf.matmul( hidden_2 , weight_output ) )+ bias_output


Learning_Rate = 0.05
cost = tf.reduce_mean(-y*tf.log(answer)-(1-y)*tf.log((1-answer)))
train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)


for i in range(100):
    print (sess.run([train, cost], feed_dict = tensor_map))
    print ("Step : ",i )


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(cost, feed_dict= tensor_map))

sess.close()