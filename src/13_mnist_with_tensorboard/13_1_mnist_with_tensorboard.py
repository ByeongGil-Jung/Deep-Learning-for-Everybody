"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-19
 Time: 오전 10:13
"""

import tensorflow as tf

from src import config
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

# Data
mnist = input_data.read_data_sets(config.DATASET["MNIST"], one_hot=True)

# Params
learning_rate = 1e-3
training_epochs = 15
batch_size = 100
batch_iter = int(mnist.train.num_examples / batch_size)
global_step = 0

TB_LOG_DIR = "./logs/mnist_1_logs"

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

tf.summary.histogram("X", X)
tf.summary.histogram("Y", Y)

# Image in Tensorboard
x_image = tf.reshape(X, [-1, 28, 28, 1])
tf.summary.image("input_image", x_image, max_outputs=3)

# Layers
with tf.variable_scope("Layer1") as scope_1:
    W1 = tf.get_variable(name="W", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([512]), name="b")
    logits1 = tf.matmul(X, W1) + b1
    L1 = tf.nn.relu(logits1)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    tf.summary.histogram("W", W1)
    tf.summary.histogram("b", b1)
    tf.summary.histogram("Layer", L1)

with tf.variable_scope("Layer2") as scope_2:
    W2 = tf.get_variable(name="W", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([512]), name="b")
    logits2 = tf.matmul(L1, W2) + b2
    L2 = tf.nn.relu(logits2)
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    tf.summary.histogram("W", W2)
    tf.summary.histogram("b", b2)
    tf.summary.histogram("Layer", L2)

with tf.variable_scope("Layer3") as scope_3:
    W3 = tf.get_variable(name="W", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([512]), name="b")
    logits3 = tf.matmul(L2, W3) + b3
    L3 = tf.nn.relu(logits3)
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

    tf.summary.histogram("W", W3)
    tf.summary.histogram("b", b3)
    tf.summary.histogram("Layer", L3)

with tf.variable_scope("Layer4") as scope_4:
    W4 = tf.get_variable(name="W", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([512]), name="b")
    logits4 = tf.matmul(L3, W4) + b4
    L4 = tf.nn.relu(logits4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

    tf.summary.histogram("W", W4)
    tf.summary.histogram("b", b4)
    tf.summary.histogram("Layer", L4)

with tf.variable_scope("Layer5") as scope_5:
    W5 = tf.get_variable(name="W", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([10]), name="b")
    logits5 = tf.matmul(L4, W5) + b5

    tf.summary.histogram("W", W5)
    tf.summary.histogram("b", b5)
    tf.summary.histogram("Hypothesis", logits5)

# Model
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits5, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

tf.summary.scalar("Loss", loss)

# Computing accuracy
prediction = tf.argmax(logits5, axis=1)
is_correct = tf.equal(prediction, tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Init the tensorboard
    merged_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter(TB_LOG_DIR)
    writer.add_graph(sess.graph)

    # Training
    print("Training started ...")
    for epoch in range(training_epochs):
        loss_avg = 0

        for i in range(batch_iter):
            x_batch, y_batch = mnist.train.next_batch(batch_size=batch_size)
            loss_val, summary, _ = sess.run(fetches=[loss, merged_summary, optimizer],
                                            feed_dict={X: x_batch, Y: y_batch, keep_prob: 0.7})

            writer.add_summary(summary=summary, global_step=global_step)

            loss_avg += loss_val / batch_iter
            global_step += 1

        print("Epoch : {:04d}, Loss : {:.9f}".format(epoch, loss_avg))

    print("Training finished ...")

    # Testing
    acc = sess.run(fetches=accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})
    print("Accuracy :", acc)

# tensorboard --logdir=./logs/mnist_1_logs


"""
Training started ...
Epoch : 0000, Loss : 0.455890183
Epoch : 0001, Loss : 0.169969561
Epoch : 0002, Loss : 0.128124239
Epoch : 0003, Loss : 0.104479300
Epoch : 0004, Loss : 0.090211551
Epoch : 0005, Loss : 0.082441536
Epoch : 0006, Loss : 0.075320073
Epoch : 0007, Loss : 0.068042951
Epoch : 0008, Loss : 0.063365634
Epoch : 0009, Loss : 0.058692478
Epoch : 0010, Loss : 0.054196861
Epoch : 0011, Loss : 0.052127332
Epoch : 0012, Loss : 0.046566512
Epoch : 0013, Loss : 0.047567675
Epoch : 0014, Loss : 0.047212779
Training finished ...

Accuracy : 0.978
"""
