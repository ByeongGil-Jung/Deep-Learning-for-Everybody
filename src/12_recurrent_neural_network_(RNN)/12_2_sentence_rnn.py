"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-05
 Time: 오후 11:17
"""

import tensorflow as tf

tf.set_random_seed(777)

# Data
sample = " If you want you"

idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}
sample_idx = [char2idx[c] for c in sample]
print("idx2char", idx2char)
print("char2idx", char2idx)
print("sample_idx", sample_idx)
"""
idx2char [' ', 'a', 'y', 'u', 'n', 'w', 't', 'I', 'f', 'o']
char2idx {' ': 0, 'a': 1, 'y': 2, 'u': 3, 'n': 4, 'w': 5, 't': 6, 'I': 7, 'f': 8, 'o': 9}
sample_idx [0, 7, 8, 0, 2, 9, 3, 0, 5, 1, 4, 6, 0, 2, 9, 3]
"""

x_data = [sample_idx[:-1]]  # hello : hell
y_data = [sample_idx[1:]]  # hello : ello
print("x_data", x_data)
print("y_data", y_data)
"""
x_data [[0, 7, 8, 0, 2, 9, 3, 0, 5, 1, 4, 6, 0, 2, 9]]
y_data [[7, 8, 0, 2, 9, 3, 0, 5, 1, 4, 6, 0, 2, 9, 3]]
"""

# Params
lr = 1e-1
imput_dim = len(idx2char)
hidden_size = len(idx2char)
num_classes = len(idx2char)
batch_size = 1
sequence_length = len(sample) - 1

X = tf.placeholder(tf.int32, shape=[None, sequence_length])
Y = tf.placeholder(tf.int32, shape=[None, sequence_length])
x_one_hot = tf.one_hot(X, depth=num_classes)
print(x_one_hot)
"""
Tensor("one_hot:0", shape=(?, 15, 10), dtype=float32)
"""

# Layers
# 1. RNN Layers
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# 2. FN Layers
# 생략

# Loss & Optimizer
weights = tf.ones([batch_size, sequence_length])
loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Prediction
prediction = tf.argmax(outputs, axis=2)

# Launch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training & Prediction
    print("step | loss | predicted Y | actual Y | predicted str")
    for step in range(51):
        loss_val, _ = sess.run(fetches=[loss, optimizer], feed_dict={X: x_data, Y: y_data})
        pred = sess.run(fetches=prediction, feed_dict={X: x_data})
        pred_str = "".join([idx2char[c] for c in pred[0]])

        print(step, loss_val, pred, y_data, pred_str)


"""
step | loss | predicted Y | actual Y | predicted str
0 2.315539 [[6 6 6 6 6 6 6 6 9 9 9 9 9 6 6]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] oooooooo     oo
1 2.2265432 [[6 6 6 6 6 6 6 6 6 6 6 6 6 6 6]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] ooooooooooooooo
2 2.1490114 [[6 6 6 6 6 6 2 9 2 9 9 9 6 6 6]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] ooooooa a   ooo
3 2.0455873 [[6 6 1 9 9 9 9 9 9 7 9 9 9 6 9]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] ooy      n   o 
4 1.9153279 [[6 9 1 1 6 9 9 9 9 7 9 9 1 6 9]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] o yyo    n  yo 
5 1.7955612 [[6 9 9 6 6 9 9 9 2 7 9 9 6 6 9]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] o  oo   an  oo 
6 1.6655123 [[6 9 9 6 6 9 9 9 2 7 9 9 6 6 9]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] o  oo   an  oo 
7 1.5779188 [[1 1 1 1 6 9 9 9 2 7 9 9 1 6 9]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] yyyyo   an  yo 
8 1.5214162 [[1 9 9 6 6 9 9 4 2 7 9 9 9 6 9]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] y  oo  wan   o 
9 1.4727715 [[1 3 9 1 6 8 9 4 2 7 5 9 1 6 9]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] yf you want yo 
10 1.4228349 [[1 3 1 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] yfyyou want you
11 1.3905218 [[1 3 1 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] yfyyou want you
12 1.3546987 [[1 3 1 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] yfyyou want you
13 1.3185765 [[1 3 1 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] yfyyou want you
14 1.2991011 [[1 3 1 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] yfyyou want you
15 1.2779181 [[0 3 1 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] Ifyyou want you
16 1.2621188 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
17 1.2384009 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
18 1.2181829 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
19 1.2009004 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
20 1.1832286 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
21 1.168514 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
22 1.1568772 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
23 1.1480958 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
24 1.1377017 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
25 1.1295621 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
26 1.12188 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
27 1.116274 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
28 1.109922 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
29 1.1050283 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
30 1.1009716 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
31 1.0974163 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
32 1.0942882 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
33 1.0917231 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
34 1.0890095 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
35 1.0865972 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
36 1.0846883 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
37 1.0828125 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
38 1.0809524 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
39 1.0792853 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
40 1.0774937 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
41 1.0756233 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
42 1.0738345 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
43 1.071905 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
44 1.0697043 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
45 1.0674678 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
46 1.0655526 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
47 1.0639668 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
48 1.0626045 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
49 1.0614673 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
50 1.060529 [[0 3 9 1 6 8 9 4 2 7 5 9 1 6 8]] [[0, 3, 9, 1, 6, 8, 9, 4, 2, 7, 5, 9, 1, 6, 8]] If you want you
"""
