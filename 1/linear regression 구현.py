#
# 결론은 cost 비용을 최소화한 값을 찾기위함임
#
# 세상의 패턴이나 특징들이 있자나 이것들은 그래프 즉 일차함수(2차함수...)로 표현할수있어
# 그 정답을 예측하고 찾아가는 과정이고
# 가설로서 기울기w 편중치b를 통하여 그값을 도출해내는역할을해
#
# 즉 minimize cost(W,b)
#
#
# import tensorflow as tf
#
# # x and Y data
# x_train =[1,2,3]
# y_train =[1,2,3]
#
# W = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')
#
# hypothesis = x_train* W + b
#
#
# #cost(W,b) = 1/m 시그마i=1~m(h(x) -y)제곱
# #cost/loss fuction
# #cost 값을 구함
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
#
#
# #minimize 최소값구함
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
#
# sess.run(tf.global_variables_initializer())
#
# for step in range(20001):
#     sess.run(train)
#     if step %20 ==0 :
#         print(step, sess.run(cost), sess.run(W), sess.run(b))
#


import tensorflow as tf

#트레이닝 데이터
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

#가설
hypothesis = X * W + b
#비용값
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#최소값
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess =tf.Session()
sess.run(tf.global_variables_initializer())

# 트레이닝 시킴
for step in range(20001) :
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X:[1,2,3,4,5,6],
                                                                         Y:[2.1,3.1,4.1,5.1,6.1,7.1]})
    if step % 20  ==0 :
        print(step, cost_val, W_val, b_val)

# hypothesis 가설에서 실제 실행
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5, 3.5]}))