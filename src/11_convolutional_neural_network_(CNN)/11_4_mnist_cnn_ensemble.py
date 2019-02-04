"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-04
 Time: 오후 5:38
"""

import tensorflow as tf
import numpy as np

from src import config
from tensorflow.examples.tutorials.mnist import input_data

# Data
mnist = input_data.read_data_sets(config.DATASET["MNIST"], one_hot=True)

# Hyper params
lr = 1e-3
training_epochs = 15
batch_size = 100
batch_iter = int(mnist.train.num_examples / batch_size)
num_models = 7  # The number of models on Ensemble


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

    def predict_with_logits(self, x_test, keep_prob=1):
        pred = self._sess.run(fetches=self._logits_last, feed_dict={self._X: x_test, self._keep_prob: keep_prob})
        return pred


# Launch
sess = tf.Session()

model_list = []
for m in range(num_models):
    model = MyMnistModel(sess=sess, name=("m" + str(m)))
    model.build_layers()
    model.build_model()
    model_list.append(model)

sess.run(tf.global_variables_initializer())

# Training
print("Training started ...")
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(model_list))

    for i in range(batch_iter):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # Training each model
        for idx, m in enumerate(model_list):
            cost_val, _ = m.train(x_train=batch_xs, y_train=batch_ys, keep_prob=0.7)
            avg_cost_list[idx] += cost_val / batch_iter

    print("Epoch : {:04d}, Cost : {}".format(epoch, avg_cost_list))

print("Training finished ...")

# Testing
pred_list = np.zeros([mnist.test.num_examples, 10])

for idx, m in enumerate(model_list):
    pred_list += m.predict_with_logits(x_test=mnist.test.images, keep_prob=1)

    print(idx, "th model's accuracy :", m.get_accuracy(x_test=mnist.test.images, y_test=mnist.test.labels, keep_prob=1))

ensemble_best_prediction = tf.argmax(pred_list, axis=1)
ensemble_is_correct = tf.equal(ensemble_best_prediction, tf.argmax(mnist.test.labels, axis=1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_is_correct, dtype=tf.float32))

print("Ensemble accuracy :", sess.run(ensemble_accuracy))


"""
Training started ...
Epoch : 0000, Cost : [0.47483257 0.4079755  0.39364978 0.39028475 0.3442808  0.34496544 0.40262776]
Epoch : 0001, Cost : [0.10391635 0.09712065 0.09763709 0.09627447 0.10271499 0.10104749 0.09132672]
Epoch : 0002, Cost : [0.07491452 0.07045848 0.07000416 0.07038544 0.07269673 0.07714267 0.06731678]
Epoch : 0003, Cost : [0.06182417 0.05952473 0.05921177 0.06045868 0.06138182 0.0617646 0.05431903]
Epoch : 0004, Cost : [0.05411042 0.05118166 0.05074648 0.0522981  0.05235652 0.05343957 0.04989523]
Epoch : 0005, Cost : [0.04893008 0.04569976 0.04530944 0.04813716 0.04746034 0.04927507 0.0424736 ]
Epoch : 0006, Cost : [0.04526163 0.04123203 0.04216338 0.04090702 0.04306793 0.04429299 0.04083832]
Epoch : 0007, Cost : [0.04012051 0.0391135  0.03670633 0.03858591 0.03907171 0.0412588 0.03590721]
Epoch : 0008, Cost : [0.03716328 0.03455518 0.03589085 0.03561666 0.03697605 0.03775993 0.03523265]
Epoch : 0009, Cost : [0.03317135 0.03603977 0.03462177 0.03403525 0.0334099  0.03516063 0.03096149]
Epoch : 0010, Cost : [0.03269942 0.03089835 0.03177685 0.03235686 0.03154636 0.03380643 0.02861482]
Epoch : 0011, Cost : [0.03259244 0.02935109 0.02889777 0.03017405 0.03030956 0.03158473 0.02931576]
Epoch : 0012, Cost : [0.02979568 0.02770259 0.02794095 0.0280337  0.02849352 0.02833521 0.02793589]
Epoch : 0013, Cost : [0.02773726 0.02749241 0.02536964 0.02688575 0.027218   0.02847083 0.02536286]
Epoch : 0014, Cost : [0.02574683 0.0267955  0.02566199 0.02578526 0.02703197 0.02710091 0.02512603]
Training finished ...

0 th model's accuracy : 0.9926
1 th model's accuracy : 0.9935
2 th model's accuracy : 0.9927
3 th model's accuracy : 0.994
4 th model's accuracy : 0.9933
5 th model's accuracy : 0.9924
6 th model's accuracy : 0.9931
Ensemble accuracy : 0.9947
"""
