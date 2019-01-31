"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-31
 Time: 오전 9:59
"""

import tensorflow as tf

tf.set_random_seed(777)

# Data (XOR)
x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_data = [[0.], [1.], [1.], [0.]]

# Params
lr = 0.1

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Making model
logits = tf.matmul(X, W) + b
hypothesis = tf.sigmoid(logits)

cost = tf.negative(tf.reduce_mean((Y * tf.log(hypothesis)) + ((1 - Y) * tf.log(1 - hypothesis))))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# Computing accuracy
prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    print("step | cost | weight")
    for step in range(10001):
        cost_val, W_val, _ = sess.run(fetches=[cost, W, optimizer], feed_dict={X: x_data, Y: y_data})

        if step % 1000 == 0:
            print(step, cost_val, W_val)

    # Accuracy
    h, c, a = sess.run(fetches=[hypothesis, prediction, accuracy], feed_dict={X: x_data, Y: y_data})
    print("Hypothesis :", h)
    print("Prediction :", c)
    print("Accuracy :", a)


"""
step | cost | weight
0 0.8875433 
[[0.7863567]
 [0.6628261]]
1000 0.69314927 
[[0.00566577]
 [0.00542595]]
2000 0.6931472 
[[0.00010906]
 [0.00010861]]
3000 0.6931472 
[[2.1310575e-06]
 [2.1321152e-06]]
4000 0.6931472 
[[1.3280558e-07]
 [1.3386332e-07]]
5000 0.6931472 
[[1.3280558e-07]
 [1.3386332e-07]]
6000 0.6931472 
[[1.3280558e-07]
 [1.3386332e-07]]
7000 0.6931472 
[[1.3280558e-07]
 [1.3386332e-07]]
8000 0.6931472 
[[1.3280558e-07]
 [1.3386332e-07]]
9000 0.6931472 
[[1.3280558e-07]
 [1.3386332e-07]]
10000 0.6931472 
[[1.3280558e-07]
 [1.3386332e-07]]
Hypothesis :
[[0.5]
 [0.5]
 [0.5]
 [0.5]]
Prediction :
[[0.]
 [0.]
 [0.]
 [0.]]
Accuracy : 0.5
"""
