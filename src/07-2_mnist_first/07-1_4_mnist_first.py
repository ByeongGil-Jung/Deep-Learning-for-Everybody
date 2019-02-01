"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-27
 Time: 오후 8:01
"""

import random
import tensorflow as tf
import matplotlib.pyplot as plt

from src import config
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

# Data
mnist = input_data.read_data_sets(config.DATASET["MNIST"], one_hot=True)

# Params
nb_classes = 10
lr = 0.1

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

# Making model
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits=logits)

cost = tf.reduce_mean(tf.negative(tf.reduce_sum(Y * tf.log(hypothesis), axis=1)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# Computing accuracy
prediction = tf.argmax(hypothesis, axis=1)
is_correct = tf.equal(prediction, tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))

# Launch
num_epochs = 15
batch_size = 100
num_batch_iter = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Learning started ...")
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_batch_iter):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size=batch_size)

            cost_val, _ = sess.run(fetches=[cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_batch_iter

        print("Epoch : {:04d}, Cost: {:.9f}".format(epoch, avg_cost))

    print("Learning finished ...")

    # Test
    print("Accuracy : {}".format(
        accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    ))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label :", sess.run(fetches=tf.argmax(mnist.test.labels[r: r + 1], axis=1)))
    print("Prediction :", sess.run(fetches=tf.argmax(hypothesis, axis=1), feed_dict={X: mnist.test.images[r: r + 1]}))

    # Show image
    plt.imshow(
        mnist.test.images[r: r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest"
    )
    plt.show()


"""
Learning started ...
Epoch : 0000, Cost: 2.826302759
Epoch : 0001, Cost: 1.061668978
Epoch : 0002, Cost: 0.838061327
Epoch : 0003, Cost: 0.733232749
Epoch : 0004, Cost: 0.669279887
Epoch : 0005, Cost: 0.624611835
Epoch : 0006, Cost: 0.591160353
Epoch : 0007, Cost: 0.563868990
Epoch : 0008, Cost: 0.541745181
Epoch : 0009, Cost: 0.522673588
Epoch : 0010, Cost: 0.506782330
Epoch : 0011, Cost: 0.492447647
Epoch : 0012, Cost: 0.479955843
Epoch : 0013, Cost: 0.468893678
Epoch : 0014, Cost: 0.458703487
Learning finished ...
Accuracy : 0.8950999975204468
Label : [1]
Prediction : [1]
"""
