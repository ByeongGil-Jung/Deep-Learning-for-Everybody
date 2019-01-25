"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-25
 Time: 오후 11:41
"""

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# Data
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Params
nb_classes = 1
lr = 1e-5

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

# Making model
hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("step | cost | pred")
    for step in range(101):
        cost_val, hy_val, _ = sess.run(fetches=[cost, hypothesis, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, hy_val)


"""
step | cost | pred
0 2455327200000.0 [[-1104436.4]
 [-2224342.8]
 [-1749606.8]
 [-1226179.4]
 [-1445287.1]
 [-1457459.5]
 [-1335740.5]
 [-1700924.6]]
1 2.69762e+27 [[3.6637149e+13]
 [7.3754336e+13]
 [5.8019879e+13]
 [4.0671629e+13]
 [4.7933685e+13]
 [4.8337135e+13]
 [4.4302659e+13]
 [5.6406091e+13]]
2 inf [[-1.2143879e+21]
 [-2.4446870e+21]
 [-1.9231472e+21]
 [-1.3481161e+21]
 [-1.5888267e+21]
 [-1.6021996e+21]
 [-1.4684714e+21]
 [-1.8696560e+21]]
3 inf [[4.0252522e+28]
 [8.1032447e+28]
 [6.3745308e+28]
 [4.4685124e+28]
 [5.2663807e+28]
 [5.3107068e+28]
 [4.8674461e+28]
 [6.1972262e+28]]
4 inf [[-1.3342243e+36]
 [-2.6859301e+36]
 [-2.1129243e+36]
 [-1.4811488e+36]
 [-1.7456130e+36]
 [-1.7603054e+36]
 [-1.6133809e+36]
 [-2.0541546e+36]]
5 inf [[inf]
 [inf]
 [inf]
 [inf]
 [inf]
 [inf]
 [inf]
 [inf]]
6 nan [[nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]]
7 nan [[nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]]

...

99 nan [[nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]]
100 nan [[nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]]
"""
