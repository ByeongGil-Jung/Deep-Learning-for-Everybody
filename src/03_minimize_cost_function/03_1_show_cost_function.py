"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-08
 Time: 오후 4:39
"""

import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)

X = [1, 2, 3]
Y = [1, 2, 3]
W = tf.placeholder(tf.float32)

hypothesis = W * X
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# List of result
W_history = []
cost_history = []

# Run
sess = tf.Session()

for i in range(-30, 50):
    curr_W = i * 0.1
    curr_cost = sess.run(cost, feed_dict={W: curr_W})

    W_history.append(curr_W)
    cost_history.append(curr_cost)

# Show cost function values by weights
plt.plot(W_history, cost_history)
plt.show()
