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
lr = 1e-2

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Making model
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.negative(tf.reduce_mean((Y * tf.log(hypothesis)) + ((1 - Y) * tf.log(1 - hypothesis))))

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
0 0.82793975
500 0.7153163
1000 0.669853
1500 0.6342829
2000 0.60624564
2500 0.58411086
3000 0.5665547
3500 0.5525306
4000 0.54122907
4500 0.53203386
5000 0.5244792
5500 0.51821303
6000 0.51296866
6500 0.5085428
7000 0.504778
7500 0.5015528
8000 0.4987709
8500 0.49635646
9000 0.49424884
9500 0.49239898
10000 0.49076667
Hypothesis : 
[[0.44348484]
 [0.9153647 ]
 [0.22591162]
 [0.93583125]
 [0.33763626]
 [0.70926887]

...

 [0.24863592]
 [0.8272391 ]
 [0.7097488 ]
 [0.7461012 ]
 [0.7991931 ]
 [0.7299595 ]
 [0.8829719 ]]
Correct : 
[[0.]
 [1.]
 [0.]
 [1.]
 [0.]
 [1.]
 [1.]

...

 [0.]
 [1.]
 [0.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]]
Accuracy :  0.7628459
"""
