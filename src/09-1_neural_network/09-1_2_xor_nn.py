"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-31
 Time: 오전 10:36
"""

import tensorflow as tf

tf.set_random_seed(777)

# Data
x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_data = [[0.], [1.], [1.], [0.]]

# Params
lr = 0.1

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])


# Making model
"""
[ Weight ]
앞의 인자는 들어오는 데이터의 갯수이며, 앞 레이어의 퍼셉트론 갯수(n) 이기도 하다.
뒤의 인자는 해당 레이어의 퍼셉트론 갯수(n)이며, 그 뒤 레이어의 각각의 퍼셉트론이 이 갯수(n)만큼 input 으로 받는다.
(output 의 갯수이기도 하다.)
"""
W1 = tf.Variable(tf.random_normal([2, 2]), name="weight1")
b1 = tf.Variable(tf.random_normal([2]), name="bias1")
logits1 = tf.matmul(X, W1) + b1
layer1 = tf.sigmoid(logits1)

W2 = tf.Variable(tf.random_normal([2, 1]), name="weight2")
b2 = tf.Variable(tf.random_normal([1]), name="bias2")
logits2 = tf.matmul(layer1, W2) + b2
hypothesis = tf.sigmoid(logits2)

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
0 0.7584403
1000 0.67128885
2000 0.5341575
3000 0.19811332
4000 0.07770608
5000 0.045380294
6000 0.031546302
7000 0.024023317
8000 0.019335365
9000 0.016148228
10000 0.013846756
Hypothesis :
[[0.00245058]
 [0.9966403 ]
 [0.99774027]
 [0.00211891]]
Prediction :
[[0.]
 [1.]
 [1.]
 [0.]]
Accuracy : 1.0
"""
