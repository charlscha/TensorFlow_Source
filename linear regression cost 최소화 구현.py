# # Lab 3 Minimizing Cost
# import tensorflow as tf
# import matplotlib.pyplot as plt
# tf.set_random_seed(777)  # for reproducibility
#
# X = [1, 2, 3]
# Y = [1, 2, 3]
#
# W = tf.placeholder(tf.float32)
#
# # Our hypothesis for linear model X * W
# hypothesis = X * W
#
# # cost/loss function
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# # Launch the graph in a session.
# sess = tf.Session()
#
# # Variables for plotting cost function
# W_history = []
# cost_history = []
#
# for i in range(-30, 50):
#     curr_W = i * 0.1
#     curr_cost = sess.run(cost, feed_dict={W: curr_W})
#     W_history.append(curr_W)
#     cost_history.append(curr_cost)
#
# # Show the cost function
# plt.plot(W_history, cost_history)
# plt.show()


# Lab 3 Minimizing Cost
import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

x_data = [1, 2, 3]
y_data = [1, 2, 3]
# Try to find values for W and b to compute y_data = W * x_data + b
# We know that W should be 1 and b should be 0
# Try to find values for W and b to compute y_data = W * x_data
# We know that W should be 1
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# # Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
# learning_rate = 0.1
# gradient = tf.reduce_mean((W * X - Y) * X)
# descent = W - learning_rate * gradient
# update = W.assign(descent)
#
# # Launch the graph in a session.
# sess = tf.Session()
# # Initializes global variables in the graph.
# sess.run(tf.global_variables_initializer())
#
# for step in range(21) :
#     sess.run(update, feed_dict={ X : x_data, Y : y_data})
#     print(step, sess.run(cost, feed_dict={ X: x_data, Y: y_data}), sess.run(W))


# Lab 3 Minimizing Cost
# This is optional
import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(5.)

# Linear model
hypothesis = X * W

# Manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
# Get gradients
gvs = optimizer.compute_gradients(cost)
gvs = optimizer.compute_gradients(cost, [W])
# Optional: modify gradient if necessary
# gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
    # Same as sess.run(train)