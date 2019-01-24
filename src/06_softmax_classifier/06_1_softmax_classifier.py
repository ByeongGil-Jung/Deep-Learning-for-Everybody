"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-24
 Time: 오후 3:39
"""

import tensorflow as tf

tf.set_random_seed(777)

# Data
x_data = [[1., 2., 1., 1.],
          [2., 1., 3., 2.],
          [3., 1., 3., 4.],
          [4., 1., 5., 5.],
          [1., 7., 5., 5.],
          [1., 2., 5., 6.],
          [1., 6., 6., 6.],
          [1., 7., 7., 7.]]

y_data = [[0., 0., 1.],
          [0., 0., 1.],
          [0., 0., 1.],
          [0., 1., 0.],
          [0., 1., 0.],
          [0., 1., 0.],
          [1., 0., 0.],
          [1., 0., 0.]]

# Params
nb_classes = 3
lr = 1e-2

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([4, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes], name="bias"))

# Making model
logits = tf.matmul(X, W) + b
hypothesis = tf.div(tf.exp(logits), tf.reduce_sum(tf.exp(logits)))
# hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.negative(tf.reduce_sum(Y * tf.log(hypothesis), axis=1)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("step | cost")
    for step in range(2001):
        cost_val, _ = sess.run(fetches=[cost, optimizer], feed_dict={X: x_data, Y: y_data})

        if step % 200 == 0:
            print(step, cost_val)

    # Testing
    print("=====================")
    r1 = sess.run(fetches=hypothesis, feed_dict={X: [[1., 11., 7., 9.]]})
    print(r1, sess.run(tf.argmax(r1, 1)))

    print("=====================")
    r2 = sess.run(fetches=hypothesis, feed_dict={X: [[1., 3., 4., 3.]]})
    print(r2, sess.run(tf.argmax(r2, 1)))

    print("=====================")
    r3 = sess.run(fetches=hypothesis, feed_dict={X: [[1., 1., 0., 1.]]})
    print(r3, sess.run(tf.argmax(r3, 1)))

    print("=====================")
    r4 = sess.run(fetches=hypothesis, feed_dict={X: [[1., 11., 7., 9.], [1., 3., 4., 3.], [1., 1., 0., 1.]]})
    print(r4, sess.run(tf.argmax(r4, 1)))


"""
step | cost
0 10.542002
200 3.5277674
400 3.1341095
600 2.9943478
800 2.902941
1000 2.8360748
1200 2.7841902
1400 2.742437
1600 2.7081385
1800 2.6796646
2000 2.6559153
=====================
[[0.4708863  0.51513124 0.01398245]] [1]
=====================
[[0.58016413 0.35284656 0.06698934]] [0]
=====================
[[0.01498721 0.07005136 0.9149614 ]] [2]
=====================
[[0.15329829 0.16770236 0.00455203]
 [0.05091285 0.03096438 0.00587871]
 [0.00879287 0.04109853 0.53679997]] [1 0 2]
"""
