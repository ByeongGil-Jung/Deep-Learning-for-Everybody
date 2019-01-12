"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-12
 Time: 오후 8:01
"""

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# Load data
xy = np.loadtxt("../../dataset/data-01-test-score.csv", delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data, "\nShape of x_data :", x_data.shape)
print(y_data, "\nShape of y_data :", y_data.shape)


"""
[[ 73.  80.  75.]
 [ 93.  88.  93.]
 [ 89.  91.  90.]

...

 [ 78.  83.  85.]
 [ 76.  83.  71.]
 [ 96.  93.  95.]]
Shape of x_data : (25, 3)
[[152.]
 [185.]
 [180.]
 
...

 [175.]
 [175.]
 [149.]
 [192.]]
Shape of y_data : (25, 1)
"""

# Parameters
lr = 1e-5

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Making model
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train = optimizer.minimize(cost)

# Fitting
sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

print("step | cost | pred")
for step in range(2001):
    cost_val, hy_val, _ = sess.run(fetches=[cost, hypothesis, train],
                                   feed_dict={X: x_data, Y: y_data})

    if step % 100 == 0:
        print(step, cost_val, hy_val)


"""
step | cost | pred
0 21027.002 [[22.048063 ]
 [21.619787 ]
 [24.096693 ]
 [22.293005 ]
 [18.633902 ]
 [ 7.2669735]
 [12.33103  ]
 [ 3.150511 ]
 [14.347944 ]
 [ 4.2534237]
 ...

...

2000 24.722479 [[154.42892]
 [185.5586 ]
 [182.90646]
 [198.08952]
 [142.52043]
 [103.55179]
 [146.7915 ]
 [106.70152]
 [172.15207]
 [157.13037]
 [142.55319]
 [140.17581]
 ...
"""

# Asking custom score
custom_score_1 = [[100., 70., 101.]]
custom_score_2 = [[60., 70., 110.],
                  [90., 100., 80.]]

pred_1 = sess.run(fetches=hypothesis, feed_dict={X: custom_score_1})
pred_2 = sess.run(fetches=hypothesis, feed_dict={X: custom_score_2})

print("Predict custom score 1 :", pred_1)
print("Predict custom score 2 :", pred_2)


"""
Predict custom score 1 : [[181.73277]]
Predict custom score 2 : [[145.86266]
 [187.2313 ]]
"""
