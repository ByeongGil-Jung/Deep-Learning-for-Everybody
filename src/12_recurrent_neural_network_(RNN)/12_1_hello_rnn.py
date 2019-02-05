"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-05
 Time: 오후 6:19
"""

import tensorflow as tf

tf.set_random_seed(777)

# Data
idx2char = ['h', 'i', 'e', 'l', 'o']  # hihell
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0]]]

y_data = [[1, 0, 2, 3, 3, 4]]  # ihello

# Params
lr = 1e-1
num_classes = 5  # 문자 집합의 총 갯수.
input_dim = 5  # one_hot 형태로 들어가는 x_data.
hidden_size = 5  # (== output_dim) 으로, 마찬가지로 one_hot 형태로 추출된다.
batch_size = 1  # 한 번에 들어가는 데이터의 양.
sequence_length = 6  # RNN 레이어의 길이이며, 여기선 예측해야 하는 문자의 총 개수이다.

X = tf.placeholder(tf.float32, shape=[None, sequence_length, input_dim])
Y = tf.placeholder(tf.int32, shape=[None, sequence_length])

# Layers
# 1. RNN Layers
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)  # RNN 로직의 엔진 역할
initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)  # RNN 을 run 시켜주는 driver 역할

# 2. FN Layers
# 생략 (원래는 있어야 함)

# Model
weights = tf.ones([batch_size, sequence_length])  # 특별한 일이 아닌 이상, 1 로 맞춰 놓을 것
loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights))  # == cost
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Prediction
prediction = tf.argmax(outputs, axis=2)

# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training & Prediction
    print("step | loss | predicted Y | actual Y | predicted str")
    for step in range(50):
        loss_val, _ = sess.run(fetches=[loss, optimizer], feed_dict={X: x_one_hot, Y: y_data})
        pred = sess.run(fetches=prediction, feed_dict={X: x_one_hot})
        pred_str = "".join([idx2char[c] for c in pred[0]])

        print(step, loss_val, pred, y_data, pred_str)


"""
step | loss | predicted Y | actual Y | predicted str
0 1.621816 [[2 2 2 2 3 3]] [[1, 0, 2, 3, 3, 4]] eeeell
1 1.5367527 [[2 2 2 3 3 3]] [[1, 0, 2, 3, 3, 4]] eeelll
2 1.4631373 [[2 2 3 3 3 3]] [[1, 0, 2, 3, 3, 4]] eellll
3 1.3964709 [[2 0 3 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehlllo
4 1.3367763 [[2 0 3 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehlllo
5 1.2807153 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
6 1.232357 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
7 1.1922058 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
8 1.1570406 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
9 1.1262147 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
10 1.0998074 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
11 1.0768883 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
12 1.0558313 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
13 1.0356215 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
14 1.0165347 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
15 0.99944067 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
16 0.9847319 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
17 0.97230387 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
18 0.9618141 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
19 0.95262414 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
20 0.9437599 [[2 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ehello
21 0.9341448 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
22 0.92322224 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
23 0.9113439 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
24 0.8993222 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
25 0.88782567 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
26 0.87723976 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
27 0.8676532 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
28 0.8587181 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
29 0.849792 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
30 0.8404842 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
31 0.8309129 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
32 0.8214522 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
33 0.81240755 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
34 0.8038915 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
35 0.7959275 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
36 0.788641 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
37 0.78223926 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
38 0.77668875 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
39 0.7715419 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
40 0.7663264 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
41 0.76101524 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
42 0.755976 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
43 0.7514951 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
44 0.7473802 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
45 0.7432153 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
46 0.7390472 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
47 0.73535013 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
48 0.73191136 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
49 0.7280953 [[1 0 2 3 3 4]] [[1, 0, 2, 3, 3, 4]] ihello
"""
