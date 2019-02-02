"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-01
 Time: 오전 11:57
"""

import tensorflow as tf
import random

from src import config
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

# Data
mnist = input_data.read_data_sets(config.DATASET["MNIST"], one_hot=True)

# Params
lr = 1e-2
training_epochs = 15
batch_size = 100
batch_iter = int(mnist.train.num_examples / batch_size)

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

# Layers
W1 = tf.Variable(tf.random_normal([784, 256]), name="weight1")
b1 = tf.Variable(tf.random_normal([256]), name="bias1")
logits1 = tf.matmul(X, W1) + b1
layer1 = tf.nn.relu(logits1)

W2 = tf.Variable(tf.random_normal([256, 256]), name="weight2")
b2 = tf.Variable(tf.random_normal([256]), name="bias2")
logits2 = tf.matmul(layer1, W2) + b2
layer2 = tf.nn.relu(logits2)

W3 = tf.Variable(tf.random_normal([256, 10]), name="weight3")
b3 = tf.Variable(tf.random_normal([10]), name="bias3")
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
    print("Learning started ...")
    for epoch in range(training_epochs):
        avg_cost = 0

        for i in range(batch_iter):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            cost_val, _ = sess.run(fetches=[cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})

            avg_cost += cost_val / batch_iter

        print("Epoch : {:04d}, Cost : {:.9f}".format(epoch, avg_cost))

    print("Learning finished ...")

    # Testing
    acc = sess.run(fetches=[accuracy], feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print("Accuracy :", acc)

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Actual label :", sess.run(fetches=tf.argmax(mnist.test.labels[r:r + 1], axis=1)))
    print("Predicted label :", sess.run(fetches=prediction, feed_dict={X: mnist.test.images[r:r + 1]}))


"""
Learning started ...
Epoch : 0000, Cost : 41.961366957
Epoch : 0001, Cost : 8.333967859
Epoch : 0002, Cost : 4.439198003
Epoch : 0003, Cost : 3.037579886
Epoch : 0004, Cost : 2.434995835
Epoch : 0005, Cost : 1.822084327
Epoch : 0006, Cost : 1.846467119
Epoch : 0007, Cost : 1.448150453
Epoch : 0008, Cost : 1.271999341
Epoch : 0009, Cost : 1.142404578
Epoch : 0010, Cost : 1.006078162
Epoch : 0011, Cost : 0.967923144
Epoch : 0012, Cost : 0.832896734
Epoch : 0013, Cost : 0.713570164
Epoch : 0014, Cost : 0.794251258
Learning finished ...
Accuracy : [0.9609]
Actual label : [4]
Predicted label : [4]
"""
