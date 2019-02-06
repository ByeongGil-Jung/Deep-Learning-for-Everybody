"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-06
 Time: 오전 1:57
"""

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# Data
sample = (" if you want to build a ship, don't drum up people together to "
          "collect wood and don't assign them tasks and work, but rather "
          "teach them to long for the endless immensity of the sea.")

char_set_to_list = list(set(sample))
char_dict = {value: index for index, value in enumerate(char_set_to_list)}
print("len(char_set_to_list)", len(char_set_to_list))  # len(char_set_to_list) 25
print("len(char_dict)", len(char_dict))  # len(char_dict) 25

# Params
lr = 1e-1
num_cells = 2
input_dim = len(char_set_to_list)
hidden_size = len(char_set_to_list)
num_classes = len(char_set_to_list)
sequence_length = 10

x_data = []
y_data = []
for i in range(0, len(sample) - sequence_length):
    x_str = sample[i:i + sequence_length]
    y_str = sample[i + 1:i + sequence_length + 1]
    print(i, x_str, "->", y_str)

    x_idx = [char_dict[c] for c in x_str]
    y_idx = [char_dict[c] for c in y_str]

    x_data.append(x_idx)
    y_data.append(y_idx)

batch_size = len(x_data)  # batch 갯수는 상관 없지만, 여기선 매우 긴 문장이 아니므로 다 넣을 것임

X = tf.placeholder(tf.int32, shape=[None, sequence_length])
Y = tf.placeholder(tf.int32, shape=[None, sequence_length])
X_one_hot = tf.one_hot(X, num_classes)

# Layers
# 1. RNN Layers
cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True) for _ in range(num_cells)]
multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(multi_cell, X_one_hot, dtype=tf.float32)  # initial_state 는 없어도 되나 ?
print(outputs)
"""
Tensor("rnn/transpose:0", shape=(?, 10, 25), dtype=float32)
"""

# 2. FN Layers
outputs = tf.reshape(outputs, shape=[-1, hidden_size])  # (-1, 25)

# logits1 = tf.contrib.layers.fully_connected(outputs, num_classes, activation_fn=None)
W1 = tf.get_variable(name="W1", shape=[25, num_classes], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([hidden_size]), name="b1")
logits1 = tf.matmul(outputs, W1) + b1

logits1 = tf.reshape(logits1, shape=[batch_size, sequence_length, num_classes])

# Loss & Optimizer
weights = tf.ones([batch_size, sequence_length])
loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=logits1, targets=Y, weights=weights))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Prediction
prediction = tf.argmax(logits1, axis=2)

# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    print("step | sequence_index | loss | predicted Y | predicted str")
    for step in range(501):
        results, loss_val, _ = sess.run(fetches=[logits1, loss, optimizer], feed_dict={X: x_data, Y: y_data})
        # results : (170, 10, 25)

        for i, result in enumerate(results):
            pred_idx = np.argmax(result, axis=1)
            pred_str = "".join([char_set_to_list[idx] for idx in pred_idx])

            if step % 100 == 0 and i % 50 == 0:
                print(step, i, loss_val, pred_idx, pred_str)

    # Check the last result
    print("\nExtracting started ...\n")
    results = sess.run(fetches=logits1, feed_dict={X: x_data})

    for i, result in enumerate(results):
        pred_idx = np.argmax(result, axis=1)

        if i is 0:
            out_str = "".join([char_set_to_list[idx] for idx in pred_idx])
            print(out_str, end='')
        else:
            print(char_set_to_list[pred_idx[-1]], end='')

    print("\nExtracting finished ...\n")


"""
0  if you wa -> if you wan
1 if you wan -> f you want
2 f you want ->  you want 
3  you want  -> you want t
4 you want t -> ou want to
5 ou want to -> u want to 

...

164 nsity of t -> sity of th
165 sity of th -> ity of the
166 ity of the -> ty of the 
167 ty of the  -> y of the s
168 y of the s ->  of the se
169  of the se -> of the sea
170 of the sea -> f the sea.
"""

"""
step | sequence_index | loss | predicted Y | predicted str
0 0 3.7262375 [3 3 3 3 3 3 3 3 3 3] rrrrrrrrrr
0 50 3.7262375 [3 3 3 3 3 3 3 3 3 3] rrrrrrrrrr
0 100 3.7262375 [3 3 3 3 3 3 3 3 3 3] rrrrrrrrrr
0 150 3.7262375 [3 3 3 3 3 3 3 3 3 3] rrrrrrrrrr
100 0 0.38494292 [15 18 23 16  7 21 23 20 12  5] tm you wan
100 50 0.38494292 [15 24 23 22 15 24 22  3 23 15] th ether t
100 100 0.38494292 [23  6 23 12  5 19 23 20  7  3]  s and wor
100 150 0.38494292 [ 3  6  5 19  8 22  6  6 23 14] rsndless i
200 0 0.23890728 [15 18 23 16  7 21 23 20 12  5] tm you wan
200 50 0.23890728 [15  7 23 22 15 24 22  3 23 15] to ether t
200 100 0.23890728 [ 6  6 23 12  5 19 23 20  7  3] ss and wor
200 150 0.23890728 [ 5 15  5 19  8 22  6  6 23 14] ntndless i
300 0 0.23355536 [15  9 23 16  7 21 23 20 12  5] tf you wan
300 50 0.23355536 [15  7 23 22 15 24 22  3 23 15] to ether t
300 100 0.23355536 [ 6  6 23 12  5 19 23 20  7  3] ss and wor
300 150 0.23355536 [ 5 15  5 19  8 22  6  6 23 14] ntndless i
400 0 0.23236056 [15  9 23 16  7 21 23 20 12  5] tf you wan
400 50 0.23236056 [15  7 23 22 15 24 22  3 23 15] to ether t
400 100 0.23236056 [ 6  6 23 12  5 19 23 20  7  3] ss and wor
400 150 0.23236056 [ 5 15  5 19  8 22  6  6 23 14] ntndless i
500 0 0.23124355 [15  9 23 16  7 21 23 20 12  5] tf you wan
500 50 0.23124355 [15  7 23 22 15 24 22  3 23 15] to ether t
500 100 0.23124355 [ 6  6 23 12  5 19 23 20  7  3] ss and wor
500 150 0.23124355 [ 3 22  5 19  8 22  6  6 23 14] rendless i

Extracting started ...

tm you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work,
but rather teach them to long for the endless immensity of the sea.

Extracting finished ...
"""
