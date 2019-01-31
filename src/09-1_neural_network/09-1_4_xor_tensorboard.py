"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-31
 Time: 오후 1:42
"""

import tensorflow as tf

tf.set_random_seed(777)

# Data
x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_data = [[0.], [1.], [1.], [0.]]

# Params
lr = 0.01

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# Making model
with tf.name_scope("Layer1"):
    W1 = tf.Variable(tf.random_normal([2, 2]), name="weight1")
    b1 = tf.Variable(tf.random_normal([2]), name="bias1")
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("Layer1", layer1)

with tf.name_scope("Layer2"):
    W2 = tf.Variable(tf.random_normal([2, 1]), name="weight2")
    b2 = tf.Variable(tf.random_normal([1]), name="bias2")
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Hypothesis", hypothesis)

with tf.name_scope("Cost"):
    cost = tf.negative(tf.reduce_mean((Y * tf.log(hypothesis)) + ((1 - Y) * tf.log(1 - hypothesis))))

    tf.summary.scalar("Cost", cost)

with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# Computing accuracy
prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
is_correct = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))

tf.summary.scalar("Accuracy", accuracy)

# Launch
with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter("./logs/xor_logs")
    writer.add_graph(sess.graph)  # Show the graph

    sess.run(tf.global_variables_initializer())

    # Training
    print("step | cost")
    for step in range(10001):
        cost_val, summary, _ = sess.run(fetches=[cost, merged_summary, optimizer], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)

        if step % 1000 == 0:
            print(step, cost_val)

    # Accuracy
    h, p, a = sess.run(fetches=[hypothesis, prediction, accuracy], feed_dict={X: x_data, Y: y_data})
    print("Hypothesis :", h)
    print("Prediction :", p)
    print("Accuracy :", a)

# tensorboard --logdir=./logs/xor_logs

"""
step | cost
0 0.7168676
1000 0.022373725
2000 0.0063154213
3000 0.0027729846
4000 0.0014256048
5000 0.00078983756
6000 0.00045537972
7000 0.00026846584
8000 0.00016036436
9000 9.6549324e-05
10000 5.838447e-05
Hypothesis :
[[6.1310318e-05]
 [9.9993694e-01]
 [9.9995077e-01]
 [5.9751477e-05]]
Prediction :
[[0.]
 [1.]
 [1.]
 [0.]]
Accuracy : 1.0
"""
