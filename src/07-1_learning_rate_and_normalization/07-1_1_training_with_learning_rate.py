"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-25
 Time: 오후 10:45
"""

import tensorflow as tf

tf.set_random_seed(777)

# Data
x_data = [[1., 2., 1.],
          [1., 3., 2.],
          [1., 3., 4.],
          [1., 5., 5.],
          [1., 7., 5.],
          [1., 2., 5.],
          [1., 6., 6.],
          [1., 7., 7.]]

y_data = [[0., 0., 1.],
          [0., 0., 1.],
          [0., 0., 1.],
          [0., 1., 0.],
          [0., 1., 0.],
          [0., 1., 0.],
          [1., 0., 0.],
          [1., 0., 0.]]

x_test = [[2., 1., 1.],
          [3., 1., 2.],
          [3., 3., 4.]]

y_test = [[0., 0., 1.],
          [0., 0., 1.],
          [0., 0., 1.]]

# Params
nb_classes = 3
lr = 0.1  # 1.5 | 1e-10 | 0.1

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([3, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

# Making model
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits=logits)

cost = tf.reduce_mean(tf.negative(tf.reduce_sum(Y * tf.log(hypothesis), axis=1)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# Computing accuracy
prediction = tf.argmax(hypothesis, axis=1)
is_correct = tf.equal(prediction, tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("> When learning_rate == {}".format(lr))
    print("step | cost | weight")
    for step in range(201):
        cost_val, W_val, _ = sess.run(fetches=[cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)

    print("Prediction :", sess.run(fetches=prediction, feed_dict={X: x_test}))
    print("Accuracy :", sess.run(fetches=accuracy, feed_dict={X: x_test, Y: y_test}))


"""
(Overfitting)
> When learning_rate == 1.5
step | cost | weight
0 5.73203 [[-0.30548954  1.2298503  -0.66033536]
 [-4.3907      2.2967086   2.9938684 ]
 [-3.345107    2.0974321  -0.80419564]]
1 23.149357 [[ 0.06951046  0.2944969  -0.0999819 ]
 [-1.9531999  -1.6362796   4.489356  ]
 [-0.9076071  -1.6502013   0.50593793]]
2 27.279778 [[ 0.44451016  0.8569968  -1.0374814 ]
 [ 0.48429942  0.9887202  -0.573143  ]
 [ 1.5298924   1.1622987  -4.7440615 ]]
3 8.668002 [[ 0.12396193  0.61504567 -0.47498202]
 [ 0.22003263 -0.2470119   0.9268558 ]
 [ 0.9603538   0.41933775 -3.431562  ]]
4 5.771106 [[-0.9524312   1.1303778   0.08607888]
 [-3.7865162   2.2624538   2.4239388 ]
 [-3.0717096   3.1403792  -2.1205401 ]]
5 inf [[nan nan nan]
 [nan nan nan]
 [nan nan nan]]
6 nan [[nan nan nan]
 [nan nan nan]
 [nan nan nan]]
7 nan [[nan nan nan]
 [nan nan nan]
 [nan nan nan]]
8 nan [[nan nan nan]
 [nan nan nan]
 [nan nan nan]]

...

199 nan [[nan nan nan]
 [nan nan nan]
 [nan nan nan]]
200 nan [[nan nan nan]
 [nan nan nan]
 [nan nan nan]]

Prediction : [0 0 0]
Accuracy : 0.0
"""

"""
(Underfitting)
> When learning_rate == 1e-10
step | cost | weight
0 5.73203 [[ 0.80269563  0.67861295 -1.2172831 ]
 [-0.3051686  -0.3032113   1.508257  ]
 [ 0.7572236  -0.7008909  -2.108204  ]]
1 5.73203 [[ 0.80269563  0.67861295 -1.2172831 ]
 [-0.3051686  -0.3032113   1.508257  ]
 [ 0.7572236  -0.7008909  -2.108204  ]]
2 5.73203 [[ 0.80269563  0.67861295 -1.2172831 ]
 [-0.3051686  -0.3032113   1.508257  ]
 [ 0.7572236  -0.7008909  -2.108204  ]]
3 5.73203 [[ 0.80269563  0.67861295 -1.2172831 ]
 [-0.3051686  -0.3032113   1.508257  ]
 [ 0.7572236  -0.7008909  -2.108204  ]]
4 5.73203 [[ 0.80269563  0.67861295 -1.2172831 ]
 [-0.3051686  -0.3032113   1.508257  ]
 [ 0.7572236  -0.7008909  -2.108204  ]]

...

198 5.73203 [[ 0.80269563  0.67861295 -1.2172831 ]
 [-0.3051686  -0.3032113   1.508257  ]
 [ 0.7572236  -0.7008909  -2.108204  ]]
199 5.73203 [[ 0.80269563  0.67861295 -1.2172831 ]
 [-0.3051686  -0.3032113   1.508257  ]
 [ 0.7572236  -0.7008909  -2.108204  ]]
200 5.73203 [[ 0.80269563  0.67861295 -1.2172831 ]
 [-0.3051686  -0.3032113   1.508257  ]
 [ 0.7572236  -0.7008909  -2.108204  ]]

Prediction : [0 0 0]
Accuracy : 0.0
"""

"""
> When learning_rate == 0.1
step | cost | weight
0 5.73203 [[ 0.7288166   0.7153621  -1.1801533 ]
 [-0.57753736 -0.12988332  1.6072978 ]
 [ 0.48373488 -0.51433605 -2.02127   ]]
1 3.317995 [[ 0.6621908   0.7479632  -1.1461285 ]
 [-0.8194891   0.03000021  1.689366  ]
 [ 0.23214608 -0.33772916 -1.9462881 ]]
2 2.0218027 [[ 0.6434202   0.74127686 -1.1206716 ]
 [-0.81161296 -0.00900121  1.7204912 ]
 [ 0.2086665  -0.3507957  -1.909742  ]]
3 1.9710886 [[ 0.6235321   0.7400824  -1.099589  ]
 [-0.8096771  -0.01636279  1.725917  ]
 [ 0.17870612 -0.33665746 -1.8939198 ]]
4 1.9446774 [[ 0.60733104  0.7366499  -1.0799555 ]
 [-0.7900784  -0.03363196  1.7235874 ]
 [ 0.16691469 -0.3340635  -1.8847224 ]]
5 1.9235673 [[ 0.59031737  0.7351024  -1.0613943 ]
 [-0.7749677  -0.0404885   1.7153332 ]
 [ 0.150766   -0.322094   -1.8805432 ]]

...

198 0.6736274 [[-1.1486679   0.28236824  1.1303256 ]
 [ 0.3735742   0.18841396  0.33788884]
 [-0.35681725 -0.43911386 -1.2559394 ]]
199 0.67226064 [[-1.1537703   0.28146935  1.1363268 ]
 [ 0.37484586  0.18958236  0.33544877]
 [-0.3560984  -0.4397301  -1.2560419 ]]
200 0.67090875 [[-1.1588541   0.28058422  1.1422957 ]
 [ 0.37609792  0.19073224  0.33304682]
 [-0.35536593 -0.44033223 -1.2561723 ]]
 
Prediction : [2 2 2]
Accuracy : 1.0
"""
