"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-16
 Time: 오후 2:57
"""

import tensorflow as tf

tf.set_random_seed(777)

# Data
x_data = [[1., 2.],
          [2., 3.],
          [3., 1.],
          [4., 3.],
          [5., 3.],
          [6., 2.]]

y_data = [[0.], [0.], [0.], [1.], [1.], [1.]]

# Parameters
lr = 1e-2

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Making model

# Sigmoid Function 을 통해 모든 값을 0 ~ 1 사이로 맞춰줌.
# (만약 0.5 이상이면 1 (==true) 를 반환하고 아니면 0 (==false) 를 반환하는 형식임)
hypothesis = tf.div(1., 1. + tf.exp(tf.negative(tf.matmul(X, W) + b)))
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.negative(tf.reduce_mean((Y * tf.log(hypothesis)) + ((1 - Y) * tf.log(1 - hypothesis))))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train = optimizer.minimize(cost)

# Computing Accuracy
# 위에서 얘기했던 것으로, H(x) 가 0.5 이상이면 1 (==true) 를 반환, 아니면 0 (==false) 를 반환
pred = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

# Launch
with tf.Session() as sess:
    # Init
    sess.run(tf.global_variables_initializer())

    # Fitting
    print("step | cost")
    for step in range(10001):
        cost_val, _ = sess.run(fetches=[cost, train],
                               feed_dict={X: x_data, Y: y_data})

        if step % 500 == 0:
            print(step, cost_val)

    # Report the accuracy
    h, c, a = sess.run(fetches=[hypothesis, pred, accuracy],
                       feed_dict={X: x_data, Y: y_data})

    print("Hypothesis : ", h)
    print("Correct : ", c)
    print("Accuracy : ", a)


"""
step | cost
0 1.7307833
500 0.48754048
1000 0.42857102
1500 0.39092326
2000 0.35999322
2500 0.33312958
3000 0.30948088
3500 0.28856295
4000 0.2699997
4500 0.25347146
5000 0.23870273
5500 0.22545676
6000 0.21353154
6500 0.20275463
7000 0.19297928
7500 0.18408065
8000 0.17595184
8500 0.16850184
9000 0.16165249
9500 0.15533635
10000 0.14949562
Hypothesis : 
[[0.03074028]
 [0.15884677]
 [0.30486727]
 [0.7813819 ]
 [0.93957496]
 [0.9801688 ]]
Correct : 
[[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]]
Accuracy :  1.0
"""
