"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-08
 Time: 오후 5:31
"""

import tensorflow as tf

tf.set_random_seed(777)

# Parameters
lr = 0.1

x_data = [1, 2, 3]
y_data = [1, 2, 3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(5.0)

# Making Model
hypothesis = W * X
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train = optimizer.minimize(cost)

# Fitting
sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

print("step | weight :")
for step in range(20):
    print(step, sess.run(W))
    sess.run(train, feed_dict={X: x_data, Y: y_data})


"""
step | weight :
0 5.0
1 1.2666664
2 1.0177778
3 1.0011852
4 1.000079
5 1.0000052
6 1.0000004
7 1.0
8 1.0
9 1.0
10 1.0
11 1.0
12 1.0
13 1.0
14 1.0
15 1.0
16 1.0
17 1.0
18 1.0
19 1.0
"""
