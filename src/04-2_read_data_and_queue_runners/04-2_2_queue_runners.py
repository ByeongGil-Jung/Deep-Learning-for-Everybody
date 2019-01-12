"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-12
 Time: 오후 8:33
"""

import tensorflow as tf

tf.set_random_seed(777)

# Load data
filename_queue = tf.train.string_input_producer(["../../dataset/data-01-test-score.csv"],  # 이 때, 다수의 파일이 들어갈 수 있음
                                                shuffle=False,
                                                name="filename_queue")

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Decoded result
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# Collect batches of csv
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]],
                                              batch_size=10)

# Parameters
lr = 1e-5

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train = optimizer.minimize(cost)

# Fitting
sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

# >> Pop the filename in 'filename_queue'
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print("step | cost | pred")
for step in range(2001):
    x_batch, y_batch = sess.run(fetches=[train_x_batch, train_y_batch])

    cost_val, hy_val, _ = sess.run(fetches=[cost, hypothesis, train],
                                   feed_dict={X: x_batch, Y: y_batch})

    if step % 100 == 0:
        print(step, cost_val, hy_val)

coord.request_stop()
coord.join(threads)


"""
step | cost | pred
0 7070.9795 [[235.22784]
 [282.40146]
 [278.39618]
 [303.91577]
 [214.62396]
 [159.15425]
 [228.5351 ]
 [170.92644]
 [264.65048]
 [246.59116]]
100 4.5778847 [[154.07788 ]
 [184.91042 ]
 [182.31238 ]
 [199.29498 ]
 [140.26404 ]
 [104.370346]
 [150.24283 ]
 [112.912056]
 [173.57487 ]
 [162.28497 ]]

...

2000 4.229059 [[153.63536 ]
 [185.07439 ]
 [182.08965 ]
 [199.36873 ]
 [140.30048 ]
 [104.96546 ]
 [150.57231 ]
 [113.519485]
 [174.27542 ]
 [163.69449 ]]
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
Predict custom score 1 : [[185.3353]]
Predict custom score 2 : [[178.36246]
 [177.03687]]
"""
