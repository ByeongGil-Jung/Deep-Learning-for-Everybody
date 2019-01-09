"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-09
 Time: 오후 9:37
"""

import tensorflow as tf

tf.set_random_seed(777)

# Parameters
lr = 1e-5

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]

y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

X = tf.placeholder(tf.float32, shape=[None, 3])  # None : data 갯수인데 상관 없음 / 3 : feature 의 갯수
Y = tf.placeholder(tf.float32, shape=[None, 1])  # None : 마찬가지로 data 갯수 / 1 : label 의 갯수 (== pred 의 갯수)

W = tf.Variable(tf.random_normal([3, 1], name="weight"))  # 3 : feature 의 갯수 / 1 : pred 의 갯수
b = tf.Variable(tf.random_normal([1], name="bias"))

# Making Model
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
0 22655.951 [[22.048063]
 [21.619787]
 [24.096693]
 [22.293005]
 [18.633902]]
100 5.64717 [[155.50876]
 [182.17552]
 [182.21854]
 [194.49936]
 [141.12082]]
200 5.449226 [[155.4331 ]
 [182.22672]
 [182.19452]
 [194.48853]
 [141.18248]]

...

1800 3.3421352 [[154.4555 ]
 [182.8865 ]
 [181.88164]
 [194.3653 ]
 [141.96141]]
1900 3.2584968 [[154.40675]
 [182.91924]
 [181.86588]
 [194.3602 ]
 [141.99907]]
2000 3.178877 [[154.3593 ]
 [182.95117]
 [181.85052]
 [194.35541]
 [142.03566]]
"""
