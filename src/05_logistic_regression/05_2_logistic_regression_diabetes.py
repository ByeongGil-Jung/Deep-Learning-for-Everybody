"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-16
 Time: 오후 4:44
"""

from src import config

import numpy as np
import tensorflow as tf

tf.set_random_seed(777)

# Read Data
xy = np.loadtxt(config.DATASET["diabetes"], delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Parameters
lr = 1e-5

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Making model
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.negative((Y * tf.log(hypothesis)) + ((1 - Y) * tf.log(1 - hypothesis)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train = optimizer.minimize(cost)

# Computing accuracy
pred = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

# Launch
with tf.Session() as sess:
    # Init
    sess.run(tf.global_variables_initializer())

    # Fitting
    print("step | cost")
    for step in range(10001):
        cost_val, _ = sess.run(fetches=[cost, train],
                               feed_dict={X: x_data, Y: y_data})

        if step % 500 == 0:
            print(step, cost_val)

    # Report accuracy
    h, c, a = sess.run(fetches=[hypothesis, pred, accuracy],
                       feed_dict={X: x_data, Y: y_data})

    print("Hypothesis : ", h)
    print("Correct : ", c)
    print("Accuracy : ", a)


"""
step | cost
0 [[0.7166428 ]
 [1.135471  ]

...

 [0.5144681 ]
 [1.2775192 ]
 [0.89986277]
 [0.9816888 ]
 [0.6557547 ]
 [0.7172558 ]
 [0.40560833]
 [0.9439044 ]]

...

10000 [[0.6145517 ]
 [0.11127948]
 [0.29984578]

...

 [1.2422698 ]
 [0.14629611]]
"""

"""
Hypothesis :  [[0.459111  ]
 [0.89469737]
 [0.25905383]
 [0.91813225]

...

 [0.7431206 ]
 [0.78580296]
 [0.7112796 ]
 [0.8639095 ]]
Correct :  [[0.]
 [1.]
 [0.]
 [1.]
 
...

 [1.]
 [1.]
 [1.]]
Accuracy :  0.7536232
"""
