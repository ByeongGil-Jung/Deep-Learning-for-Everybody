"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-31
 Time: 오전 11:12
"""

import tensorflow as tf

tf.set_random_seed(777)

# Data
x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_data = [[0.], [1.], [1.], [0.]]

# Params
lr = 0.1

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

# Making model
W1 = tf.Variable(tf.random_normal([2, 10]), name="weight1")
b1 = tf.Variable(tf.random_normal([10]), name="bias1")
logits1 = tf.matmul(X, W1) + b1
layer1 = tf.sigmoid(logits1)

W2 = tf.Variable(tf.random_normal([10, 10]), name="weight2")
b2 = tf.Variable(tf.random_normal([10]), name="bias2")
logits2 = tf.matmul(layer1, W2) + b2
layer2 = tf.sigmoid(logits2)

W3 = tf.Variable(tf.random_normal([10, 10]), name="weight3")
b3 = tf.Variable(tf.random_normal([10]), name="bias3")
logits3 = tf.matmul(layer2, W3) + b3
layer3 = tf.sigmoid(logits3)

W4 = tf.Variable(tf.random_normal([10, 1]), name="weight4")
b4 = tf.Variable(tf.random_normal([1]), name="bias4")
logits4 = tf.matmul(layer3, W4) + b4
hypothesis = tf.sigmoid(logits4)

cost = tf.negative(tf.reduce_mean((Y * tf.log(hypothesis)) + ((1 - Y) * tf.log(1 - hypothesis))))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# Computing accuracy
prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
is_correct = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))

# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    print("step | cost")
    for step in range(10001):
        cost_val, _ = sess.run(fetches=[cost, optimizer], feed_dict={X: x_data, Y: y_data})

        if step % 1000 == 0:
            print(step, cost_val)

    # Accuracy
    h, p, a = sess.run(fetches=[hypothesis, prediction, accuracy], feed_dict={X: x_data, Y: y_data})
    print("Hypothesis :", h)
    print("Prediction :", p)
    print("Accuracy :", a)


"""
step | cost
0 0.7246195
1000 0.6472316
2000 0.11493275
3000 0.014284084
4000 0.0062706145
5000 0.0038306066
6000 0.0027027968
7000 0.0020658378
8000 0.001660981
9000 0.0013828698
10000 0.0011808933
Hypothesis :
[[1.3571062e-04]
 [9.9987793e-01]
 [9.9972171e-01]
 [2.5818727e-04]]
Prediction :
[[0.]
 [1.]
 [1.]
 [0.]]
Accuracy : 1.0
"""
