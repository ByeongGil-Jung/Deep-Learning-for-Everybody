"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-01
 Time: 오후 4:57
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

# Layers
W1 = tf.get_variable(name="W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]), name="b1")
logits1 = tf.matmul(X, W1) + b1
layer1 = tf.nn.relu(logits1)

W2 = tf.get_variable(name="W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]), name="b2")
logits2 = tf.matmul(layer1, W2) + b2
layer2 = tf.nn.relu(logits2)

W3 = tf.get_variable(name="W3", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]), name="b3")
logits3 = tf.matmul(layer2, W3) + b3

# Model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits3, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# Computing accuracy
prediction = tf.argmax(logits3, axis=1)
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
Epoch : 0000, Cost : 0.770926981
Epoch : 0001, Cost : 0.292452535
Epoch : 0002, Cost : 0.231891093
Epoch : 0003, Cost : 0.194845273
Epoch : 0004, Cost : 0.167942947
Epoch : 0005, Cost : 0.147097695
Epoch : 0006, Cost : 0.129834544
Epoch : 0007, Cost : 0.116537202
Epoch : 0008, Cost : 0.104516705
Epoch : 0009, Cost : 0.094808159
Epoch : 0010, Cost : 0.086176658
Epoch : 0011, Cost : 0.078119622
Epoch : 0012, Cost : 0.071820867
Epoch : 0013, Cost : 0.065954007
Epoch : 0014, Cost : 0.060457226
Training finished ...
Accuracy : 0.9725
Actual label : [3]
Predicted label : [3]
"""
