"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-04
 Time: 오후 4:22
"""

import tensorflow as tf

from src import config
from tensorflow.examples.tutorials.mnist import input_data

# Data
mnist = input_data.read_data_sets(config.DATASET["MNIST"], one_hot=True)

# Hyper params
lr = 1e-3
training_epochs = 15
batch_size = 100
batch_iter = int(mnist.train.num_examples / batch_size)


# Model Class
class MyMnistModel(object):
    def __init__(self, sess, name):
        self._sess = sess
        self._name = name

        self._X = tf.placeholder(tf.float32, shape=[None, 784])
        self._Y = tf.placeholder(tf.float32, shape=[None, 10])
        self._keep_prob = tf.placeholder(tf.float32)

        self._logits_last = None
        self._cost = None
        self._optimizer = None
        self._prediction = None
        self._is_correct = None
        self._accuracy = None

    def build_layers(self):
        with tf.variable_scope(self._name):
            X_img = tf.reshape(self._X, shape=[-1, 28, 28, 1])

            # Layers
            # 1. CNN Layers
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))  # (?, 28, 28, 1)
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")  # (?, 28, 28, 32)
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # (?, 14, 14, 32)
            L1 = tf.nn.dropout(L1, keep_prob=self._keep_prob)

            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))  # (?, 14, 14, 32)
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")  # (?, 14, 14, 64)
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # (?, 7, 7, 64)
            L2 = tf.nn.dropout(L2, keep_prob=self._keep_prob)

            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))  # (?, 7, 7, 64)
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding="SAME")  # (?, 7, 7, 128)
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # (?, 4, 4, 128)
            L3 = tf.nn.dropout(L3, keep_prob=self._keep_prob)

            # 2. FN Layers
            L3_flat = tf.reshape(L3, shape=[-1, 4 * 4 * 128])  # (?, 2048)
            W4 = tf.get_variable(name="W4", shape=[2048, 625], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]), name="b4")
            logits4 = tf.matmul(L3_flat, W4) + b4
            L4 = tf.nn.relu(logits4)
            L4 = tf.nn.dropout(L4, keep_prob=self._keep_prob)

            W5 = tf.get_variable(name="W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]), name="b5")
            self._logits_last = tf.matmul(L4, W5) + b5

    def build_model(self):
        # Cost & Optimizer
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._logits_last, labels=self._Y))
        self._optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._cost)

        # Prediction & Accuracy
        self._prediction = tf.argmax(self._logits_last, axis=1)
        self._is_correct = tf.equal(self._prediction, tf.argmax(self._Y, axis=1))
        self._accuracy = tf.reduce_mean(tf.cast(self._is_correct, dtype=tf.float32))

    def train(self, x_train, y_train, keep_prob=0.7):
        c_val, _ = self._sess.run(fetches=[self._cost, self._optimizer],
                                  feed_dict={self._X: x_train, self._Y: y_train, self._keep_prob: keep_prob})
        return c_val, _

    def get_accuracy(self, x_test, y_test, keep_prob=1):
        acc = self._sess.run(fetches=self._accuracy,
                             feed_dict={self._X: x_test, self._Y: y_test, self._keep_prob: keep_prob})
        return acc

    def predict(self, x_test, keep_prob=1):
        pred = self._sess.run(fetches=self._prediction, feed_dict={self._X: x_test, self._keep_prob: keep_prob})
        return pred


# Launch
sess = tf.Session()

m1 = MyMnistModel(sess=sess, name="m1")
m1.build_layers()
m1.build_model()

sess.run(tf.global_variables_initializer())

# Training
print("Training started ...")
for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(batch_iter):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        cost_val, _ = m1.train(x_train=batch_xs, y_train=batch_ys, keep_prob=0.7)

        avg_cost += cost_val / batch_iter

    print("Epoch : {:04d}, Cost : {:.9f}".format(epoch, avg_cost))

print("Training finished ...")

# Testing
acc = m1.get_accuracy(x_test=mnist.test.images, y_test=mnist.test.labels, keep_prob=1)
print("Accuracy :", acc)


"""
Training started ...
Epoch : 0000, Cost : 0.358666966
Epoch : 0001, Cost : 0.093875589
Epoch : 0002, Cost : 0.071904059
Epoch : 0003, Cost : 0.058962016
Epoch : 0004, Cost : 0.051093624
Epoch : 0005, Cost : 0.048415190
Epoch : 0006, Cost : 0.041574690
Epoch : 0007, Cost : 0.038841079
Epoch : 0008, Cost : 0.036289163
Epoch : 0009, Cost : 0.035019322
Epoch : 0010, Cost : 0.031736554
Epoch : 0011, Cost : 0.029541511
Epoch : 0012, Cost : 0.028451967
Epoch : 0013, Cost : 0.028020192
Epoch : 0014, Cost : 0.027621494
Training finished ...
Accuracy : 0.9939
"""
