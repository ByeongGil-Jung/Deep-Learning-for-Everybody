"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-24
 Time: 오후 4:45
"""

import tensorflow as tf
import numpy as np

from src import config

tf.set_random_seed(777)

# Data
xy = np.loadtxt(config.DATASET["zoo"], delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)
"""
(101, 16) (101, 1)
"""

# Params
nb_classes = 7
lr = 1e-2

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])

W = tf.Variable(tf.random_normal([16, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

# One-hot encoding
Y_one_hot = tf.one_hot(Y, nb_classes)  # 만약 input 의 rank 가 n 이라면, output 의 rank 는 n + 1 이 됨 (그래서 밑에서 reshape 함)
print("one_hot :", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, shape=[-1, nb_classes])  # 앞의 -1 은 모든 값을 의미하는 것으로, nb_classes 갯수의 리스트들로 바꾼다.
print("reshaped_one_hot :", Y_one_hot)
"""
one_hot : Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshaped_one_hot : Tensor("Reshape:0", shape=(?, 7), dtype=float32)
"""

# Making model
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# Computing accuracy
prediction = tf.argmax(hypothesis, axis=1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("cost | accuracy")
    for step in range(2001):
        cost_val, acc_val, _ = sess.run(fetches=[cost, accuracy, optimizer], feed_dict={X: x_data, Y: y_data})

        if step % 100 == 0:
            print(cost_val, acc_val)

    # Prediction
    print("== Prediction ==")
    pred = sess.run(fetches=prediction, feed_dict={X: x_data})

    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Pred Y : {}, True Y : {}".format(p == int(y), p, int(y)))


"""
cost | accuracy
[ 7.8791256   5.930115    5.65278     7.8791256   6.2693067   5.930115
  5.8019037   5.840953    5.65278     8.555001    6.2693067   9.60522
  5.65278     1.054547    3.79913     6.99302    10.127355    5.930115
  7.468528    3.2592125   9.60522    10.753661    5.930115    9.438375
 12.765127    1.5509738   2.9602468   6.5875793   5.930115    5.686671
 11.122916    5.8019037   5.325334   12.031051    5.8997235   6.2301993
  6.4305196  10.127355    5.65278    11.726457    9.31074    11.617855
 11.177262    8.822781    6.2693067   6.2693067   6.99302     6.2693067
  6.261009    6.7093954   6.2693067   9.31074     0.8805712   9.434945
  6.7093954   5.930115   10.9099455   9.60522    14.154967    8.822781
  7.468528    5.65278     3.0542557   3.299665    6.2693067   5.8019037
  3.2592125   6.2693067   6.205142    6.2693067   5.8019037  12.307193
 11.936497    5.8997235   2.8909755   3.49726     1.3729575   0.6429189
 12.031051   12.031051    2.7828255   0.56990385  5.8997235   8.822781
  4.1697392   5.3268695   6.894843   11.3719225  12.765127    1.7860607
  3.1267395   2.7754972   7.468528    6.5875793   6.4305196  10.798417
  4.107725   10.794503    6.2693067   0.56990385  8.822781  ] 0.03960396
[2.1766230e-02 3.3444208e-03 3.5091188e-02 2.1766230e-02 4.9231262e-03
 3.3444208e-03 1.4493687e-03 2.7130207e-02 3.5091188e-02 4.2831924e-02
 4.9231262e-03 1.5354053e-02 3.5091188e-02 1.4378433e-01 8.0573209e-02
 1.2978058e-01 1.9537274e-02 3.3444208e-03 8.2997113e-02 9.3641289e-02
 1.5354053e-02 3.3728410e-02 3.3444208e-03 8.7571628e-03 3.8074484e-01
 1.2593994e-01 4.9887884e-01 7.4554019e-02 3.3444208e-03 4.9747261e-03
 3.9416209e-02 1.4493687e-03 5.3127296e-03 4.6511535e-02 3.9692815e-02
 5.6372238e-03 1.2651306e-02 1.9537274e-02 3.5091188e-02 2.9336996e-02
 8.6907484e-03 2.0592657e-01 3.5438165e-01 1.4510346e-02 4.9231262e-03
 4.9231262e-03 1.2978058e-01 4.9231262e-03 7.8458944e-03 1.7136013e-02
 4.9231262e-03 8.6907484e-03 6.0261679e-01 1.9519690e-01 1.7136013e-02
 3.3444208e-03 8.6530045e-02 1.5354053e-02 3.8493118e-01 1.4510346e-02
 8.2997113e-02 3.5091188e-02 2.0810150e-01 2.0744503e-02 4.9231262e-03
 1.4493687e-03 9.3641289e-02 4.9231262e-03 2.2501643e-03 4.9231262e-03
 1.4493687e-03 1.2618344e-01 6.8445081e-01 3.9692815e-02 1.0435539e-02
 1.3530633e-03 4.8231673e-01 3.1485677e-02 4.6511535e-02 4.6511535e-02
 4.7532052e-01 6.3899553e-01 3.9692815e-02 1.4510346e-02 5.9297360e-03
 1.0010262e-01 1.4279953e-01 1.8551180e-02 3.8074484e-01 1.3147120e-01
 1.6154320e+00 3.4998375e-01 8.2997113e-02 7.4554019e-02 1.2651306e-02
 1.2368524e-02 1.7942773e-03 6.2368996e-02 4.9231262e-03 6.3899553e-01
 1.4510346e-02] 0.990099
 
...
 
[2.8451209e-04 2.7187943e-04 1.7615530e-03 2.8451209e-04 2.7569308e-04
 2.7187943e-04 7.9151832e-05 8.5245981e-04 1.7615530e-03 2.1127779e-03
 2.7569308e-04 5.9169903e-04 1.7615530e-03 3.0236978e-03 2.8043964e-03
 9.4000362e-03 2.6222604e-04 2.7187943e-04 1.9396793e-03 4.8927576e-03
 5.9169903e-04 5.9860904e-04 2.7187943e-04 6.4292742e-04 2.4944467e-02
 1.9151694e-03 1.8237991e-02 2.4195225e-03 2.7187943e-04 4.4344873e-05
 1.1312322e-04 7.9151832e-05 1.0477947e-04 2.8093683e-04 1.3205627e-03
 2.0990553e-04 6.6043978e-04 2.6222604e-04 1.7615530e-03 4.4741156e-04
 1.7762026e-05 1.0024087e-02 2.4308836e-02 7.5478671e-04 2.7569308e-04
 2.7569308e-04 9.4000362e-03 2.7569308e-04 1.9393470e-04 6.5019447e-04
 2.7569308e-04 1.7762026e-05 6.5922864e-02 1.8971303e-02 6.5019447e-04
 2.7187943e-04 1.7010268e-02 5.9169903e-04 8.6747948e-03 7.5478671e-04
 1.9396793e-03 1.7615530e-03 1.4782940e-03 4.1438197e-03 2.7569308e-04
 7.9151832e-05 4.8927576e-03 2.7569308e-04 8.0463033e-05 2.7569308e-04
 7.9151832e-05 7.7308035e-03 3.6896262e-02 1.3205627e-03 2.2253898e-04
 1.1444026e-05 4.7873605e-02 2.1050144e-04 2.8093683e-04 2.8093683e-04
 1.4371108e-02 3.9064772e-02 1.3205627e-03 7.5478671e-04 5.9789419e-04
 4.6819351e-03 1.7745066e-02 1.5090757e-04 2.4944467e-02 2.2774017e-03
 5.6622185e-02 4.8892863e-02 1.9396793e-03 2.4195225e-03 6.6043978e-04
 2.5054652e-04 2.8391622e-04 1.2705596e-03 2.7569308e-04 3.9064772e-02
 7.5478671e-04] 1.0
 
== Prediction ==
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 6, True Y : 6
[True] Pred Y : 6, True Y : 6
[True] Pred Y : 6, True Y : 6
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 5, True Y : 5
[True] Pred Y : 4, True Y : 4
[True] Pred Y : 4, True Y : 4
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 5, True Y : 5
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 5, True Y : 5
[True] Pred Y : 5, True Y : 5
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 5, True Y : 5
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 6, True Y : 6
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 5, True Y : 5
[True] Pred Y : 4, True Y : 4
[True] Pred Y : 6, True Y : 6
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 2, True Y : 2
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 6, True Y : 6
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 2, True Y : 2
[True] Pred Y : 6, True Y : 6
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 2, True Y : 2
[True] Pred Y : 6, True Y : 6
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 6, True Y : 6
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 5, True Y : 5
[True] Pred Y : 4, True Y : 4
[True] Pred Y : 2, True Y : 2
[True] Pred Y : 2, True Y : 2
[True] Pred Y : 3, True Y : 3
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 1, True Y : 1
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 5, True Y : 5
[True] Pred Y : 0, True Y : 0
[True] Pred Y : 6, True Y : 6
[True] Pred Y : 1, True Y : 1
"""
