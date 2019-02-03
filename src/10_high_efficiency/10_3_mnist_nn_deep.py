"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-02
 Time: 오후 10:27
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
Y = tf.placeholder(tf.float32, shape=[None, 10])

# Layers
W1 = tf.get_variable(name="W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]), name="b1")
logits1 = tf.matmul(X, W1) + b1
layer1 = tf.nn.relu(logits1)

W2 = tf.get_variable(name="W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]), name="b2")
logits2 = tf.matmul(layer1, W2) + b2
layer2 = tf.nn.relu(logits2)

W3 = tf.get_variable(name="W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]), name="b3")
logits3 = tf.matmul(layer2, W3) + b3
layer3 = tf.nn.relu(logits3)

W4 = tf.get_variable(name="W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]), name="b4")
logits4 = tf.matmul(layer3, W4) + b4
layer4 = tf.nn.relu(logits4)

W5 = tf.get_variable(name="W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]), name="b5")
logits5 = tf.matmul(layer4, W5) + b5

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
            cost_val, _ = sess.run(fetches=[cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})

            avg_cost += cost_val / batch_iter

        print("Epoch : {:04d}, Cost : {:.9f}".format(epoch, avg_cost))

    print("Training finished ...")

    # Testing
    acc = sess.run(fetches=accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print("Accuracy :", acc)

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Actual label :", sess.run(fetches=tf.argmax(mnist.test.labels[r:r + 1], axis=1)))
    print("Predicted label :", sess.run(fetches=prediction, feed_dict={X: mnist.test.images[r:r + 1]}))


"""
Training started ...
Epoch : 0000, Cost : 0.296692585
Epoch : 0001, Cost : 0.105126909
Epoch : 0002, Cost : 0.070916030
Epoch : 0003, Cost : 0.054094118
Epoch : 0004, Cost : 0.041283473
Epoch : 0005, Cost : 0.035529787
Epoch : 0006, Cost : 0.028978590
Epoch : 0007, Cost : 0.025732956
Epoch : 0008, Cost : 0.025104730
Epoch : 0009, Cost : 0.018081026
Epoch : 0010, Cost : 0.020960107
Epoch : 0011, Cost : 0.017482232
Epoch : 0012, Cost : 0.015934724
Epoch : 0013, Cost : 0.016768658
Epoch : 0014, Cost : 0.011958182
Training finished ...
Accuracy : 0.9787
Actual label : [6]
Predicted label : [6]
"""
