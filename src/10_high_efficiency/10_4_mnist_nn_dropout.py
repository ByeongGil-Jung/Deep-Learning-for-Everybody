"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-02
 Time: 오후 11:24
"""

import tensorflow as tf
import random

from src import config
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

# Data
mnist = input_data.read_data_sets(config.DATASET["MNIST"], one_hot=True)

# Params
lr = 1e-4
training_epochs = 15
batch_size = 100
batch_iter = int(mnist.train.num_examples / batch_size)

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# Layers
W1 = tf.get_variable(name="W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]), name="b1")
logits1 = tf.matmul(X, W1) + b1
L1 = tf.nn.relu(logits1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable(name="W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]), name="b2")
logits2 = tf.matmul(L1, W2) + b2
L2 = tf.nn.relu(logits2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable(name="W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]), name="b3")
logits3 = tf.matmul(L2, W3) + b3
L3 = tf.nn.relu(logits3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable(name="W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]), name="b4")
logits4 = tf.matmul(L3, W4) + b4
L4 = tf.nn.relu(logits4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable(name="W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]), name="b5")
logits5 = tf.matmul(L4, W5) + b5

# Model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits5, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# Computing accuracy
prediction = tf.argmax(logits5, axis=1)
is_correct = tf.equal(prediction, tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))

# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    print("Training started ...")
    for epoch in range(training_epochs):
        avg_cost = 0

        for i in range(batch_iter):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            cost_val, _ = sess.run(fetches=[cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})

            avg_cost += cost_val / batch_iter

        print("Epoch : {:04d}, Cost : {:.9f}".format(epoch, avg_cost))

    print("Training finished ...")

    # Testing
    acc = sess.run(fetches=accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})
    print("Accuracy :", acc)

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Actual label :", sess.run(fetches=tf.argmax(mnist.test.labels[r:r + 1], axis=1)))
    print("Predicted label :", sess.run(fetches=prediction, feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))


"""
Training started ...
Epoch : 0000, Cost : 1.266917695
Epoch : 0001, Cost : 0.435104661
Epoch : 0002, Cost : 0.318633587
Epoch : 0003, Cost : 0.253874611
Epoch : 0004, Cost : 0.214179432
Epoch : 0005, Cost : 0.189089316
Epoch : 0006, Cost : 0.167352054
Epoch : 0007, Cost : 0.150042513
Epoch : 0008, Cost : 0.134940860
Epoch : 0009, Cost : 0.125804711
Epoch : 0010, Cost : 0.115580746
Epoch : 0011, Cost : 0.107408731
Epoch : 0012, Cost : 0.100358585
Epoch : 0013, Cost : 0.093494002
Epoch : 0014, Cost : 0.086824633
Training finished ...
Accuracy : 0.9761
Actual label : [2]
Predicted label : [2]
"""
