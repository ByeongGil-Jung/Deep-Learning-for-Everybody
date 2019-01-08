"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-08
 Time: 오후 5:03
"""

import tensorflow as tf

tf.set_random_seed(777)

# Parameters
lr = 0.1
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name="weight")

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Making cost function
hypothesis = W * X
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Making GD
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - lr * gradient
update = W.assign(descent)

# Fitting
sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

print("step | curr_cost | weight :")
for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})

    curr_cost = sess.run(cost, feed_dict={X: x_data, Y: y_data})
    print(step, curr_cost, sess.run(W))


"""
step | curr_cost | weight :
0 1.9391857 [1.6446238]
1 0.5515905 [1.3437994]
2 0.15689684 [1.1833596]
3 0.044628453 [1.0977918]
4 0.012694317 [1.0521556]
5 0.003610816 [1.0278163]
6 0.0010270766 [1.0148354]
7 0.00029214387 [1.0079122]
8 8.309683e-05 [1.0042198]
9 2.363606e-05 [1.0022506]
10 6.723852e-06 [1.0012003]
11 1.912386e-06 [1.0006402]
12 5.439676e-07 [1.0003414]
13 1.5459062e-07 [1.000182]
14 4.3941593e-08 [1.000097]
15 1.2491266e-08 [1.0000517]
16 3.5321979e-09 [1.0000275]
17 9.998237e-10 [1.0000147]
18 2.8887825e-10 [1.0000079]
19 8.02487e-11 [1.0000042]
20 2.3405278e-11 [1.0000023]
"""
