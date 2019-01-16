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
cost = tf.negative((Y * tf.log(hypothesis)) + ((1 - Y) * tf.log(1 - hypothesis)))

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
0 [[2.6887839e+00]
 [4.1161499e+00]
 [3.5736346e+00]
 [3.3234088e-03]
 [1.4906902e-03]
 [1.3168777e-03]]
500 [[0.12495273]
 [0.2340797 ]
 [0.88618463]
 [0.40180975]
 [0.16432585]
 [0.04512734]]

...

9500 [[0.00103569]
 [0.05333764]
 [0.06963457]
 [0.07379037]
 [0.00494814]
 [0.00109971]]
10000 [[0.00092352]
 [0.05106884]
 [0.06626654]
 [0.07070091]
 [0.00452863]
 [0.00098754]]
"""

"""
Hypothesis :  [[9.2288991e-04]
 [4.9782604e-02]
 [6.4112566e-02]
 [9.3174601e-01]
 [9.9548244e-01]
 [9.9901319e-01]]
Correct :  [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]]
Accuracy :  1.0
"""
