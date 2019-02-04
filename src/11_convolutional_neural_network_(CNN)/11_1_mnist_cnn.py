"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-03
 Time: 오후 8:00
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
X_img = tf.reshape(X, shape=[-1, 28, 28, 1])  # -1 : 모든 데이터 | 28, 28 : 28x28 행렬 | 1 : grey
Y = tf.placeholder(tf.float32, shape=[None, 10])

# Layers
# 1. Convolution Layers
"""
< Weight >
 [shape] 
 (a, b, c, d) -> a, b : axb 행렬 | c : channel 갯수 | d : filter 갯수
 
 ==============
 
 > 3, 3 : 3x3 행렬 | 1 : channel == depth == color(1) | 32 개의 filter

--------------------------------------------------------------------------------------------------------

< conv2d >
 [strides]
 (a, b, c, d) -> a : batch size | b, c, d : bxcxd 행렬 (여기서 d 는 channel 의 갯수)
 
 > 1 : batch size | 1, 1 : 1x1 행렬 | 1 : channel == depth == color(1)
 
 ==============
 
 [padding="SAME"] 을 통해서 conv 하기 전에 주변에 0 을 채워넣어서, conv 후의 모양과 하기 전의 모양이 같도록 만듦

--------------------------------------------------------------------------------------------------------

< max_pool >
 [ksize] : pooling 할 행렬의 모양
 (a, b, c, d) -> a : batch size | b, c, d : bxcxd 행렬 (여기서 d 는 channel 의 갯수)
 
 > 1 : batch_size | 2, 2 : 2x2 행렬 | 1 : channel == depth == color(1)
 
 ==============
 
 [strides] : pooling 을 얼만큼 어떻게 할 것인가에 대한 것
 (a, b, c, d) -> a : batch size | b, c, d : bxcxd 행렬 (여기서 d 는 channel 의 갯수)
 
 > 1 : batch_size | 2, 2 : 2x2 행렬 | 1 : channel == depth == color(1)
 
 ==============
 
 [padding="SAME"] 을 통해서 conv 하기 전에 주변에 0 을 채워넣어서, conv 후의 모양과 하기 전의 모양이 같도록 만듦
 
"""
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))  # Img shape : (?, 28, 28, 1)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")  # Conv shape : (?, 28, 28, 32)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # Pooled shape : (?, 14, 14, 32)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))  # Img shape : (?, 14, 14, 32)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")  # Conv shape : (?, 14, 14, 64)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # Pooled shape : (?, 7, 7, 64)

# 2. Fully Connected Layers
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])  # Reshaped shape : (?, 3136)
W3 = tf.get_variable(name="W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]), name="bias")
logits = tf.matmul(L2_flat, W3) + b

# Model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# Computing accuracy
prediction = tf.argmax(logits, axis=1)
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
Epoch : 0000, Cost : 0.345738340
Epoch : 0001, Cost : 0.091908623
Epoch : 0002, Cost : 0.068224209
Epoch : 0003, Cost : 0.056426266
Epoch : 0004, Cost : 0.046828212
Epoch : 0005, Cost : 0.041030964
Epoch : 0006, Cost : 0.036459860
Epoch : 0007, Cost : 0.032386299
Epoch : 0008, Cost : 0.027609222
Epoch : 0009, Cost : 0.024448187
Epoch : 0010, Cost : 0.021894022
Epoch : 0011, Cost : 0.020112594
Epoch : 0012, Cost : 0.016673039
Epoch : 0013, Cost : 0.015265015
Epoch : 0014, Cost : 0.013137653
Training finished ...
Accuracy : 0.9889
Actual label : [7]
Predicted label : [7]
"""
