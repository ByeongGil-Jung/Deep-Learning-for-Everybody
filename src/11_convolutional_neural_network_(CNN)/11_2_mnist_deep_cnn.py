"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-04
 Time: 오후 2:05
"""

import tensorflow as tf
import random

from src import config
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

# Data
mnist = input_data.read_data_sets(config.DATASET["MNIST"], one_hot=True)

# Params
lr = 1e-3
training_epochs = 15
batch_size = 100
batch_iter = int(mnist.train.num_examples / batch_size)

X = tf.placeholder(tf.float32, shape=[None, 784])
X_img = tf.reshape(X, shape=[-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# Layers
# 1. CNN Layers
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))  # Img shape : (?, 28, 28, 1)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")  # Conv shape : (?, 28, 28, 32)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # Pooled shape : (?, 14, 14, 32)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))  # Img shape : (?, 14, 14, 32)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")  # Conv shape : (?, 14, 14, 64)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # Pooled shape : (?, 7, 7, 64)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))  # Img shape : (?, 7, 7, 64)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding="SAME")  # Conv shape : (?, 7, 7, 128)
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # Pooled shape : (?, 4, 4, 128)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# 2. FN Layers
L3_flat = tf.reshape(L3, shape=[-1, 4 * 4 * 128])  # Reshaped shape : (?, 2048)
W4 = tf.get_variable(name="W4", shape=[2048, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]), name="b4")
logits4 = tf.matmul(L3_flat, W4) + b4
L4 = tf.nn.relu(logits4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable(name="W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
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
Epoch : 0000, Cost : 0.370626152
Epoch : 0001, Cost : 0.099693451
Epoch : 0002, Cost : 0.073919126
Epoch : 0003, Cost : 0.061203060
Epoch : 0004, Cost : 0.054754952
Epoch : 0005, Cost : 0.047730092
Epoch : 0006, Cost : 0.044047536
Epoch : 0007, Cost : 0.039717223
Epoch : 0008, Cost : 0.037068700
Epoch : 0009, Cost : 0.034950825
Epoch : 0010, Cost : 0.034243008
Epoch : 0011, Cost : 0.030323614
Epoch : 0012, Cost : 0.028478175
Epoch : 0013, Cost : 0.026927626
Epoch : 0014, Cost : 0.027117396
Training finished ...
Accuracy : 0.9942
Actual label : [7]
Predicted label : [7]
"""
