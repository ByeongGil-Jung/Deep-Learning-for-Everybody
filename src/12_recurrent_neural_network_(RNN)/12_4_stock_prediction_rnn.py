"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-06
 Time: 오후 3:45
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from src import config

tf.set_random_seed(777)


"""
Params
"""
lr = 1e-2
seq_length = 7
batch_size = 300
input_dim = 5
hidden_dim = 10
output_dim = 1
training_iter = 3000

X = tf.placeholder(tf.float32, shape=[None, seq_length, input_dim])  # (-1, 7, 5)
Y = tf.placeholder(tf.float32, shape=[None, output_dim])  # (-1, 1)


"""
Data
"""
# Load data
xy = np.loadtxt(config.DATASET["stock_daily"], delimiter=',')
xy = xy[::-1]  # 가장 최근의 값 순으로 나열

# Normalization
scaler = MinMaxScaler()
xy = scaler.fit_transform(xy)

# Split data to X and Y
X_data = []
y_data = []
for i in range(0, len(xy) - seq_length):
    _x = xy[i:i + seq_length, :]
    _y = xy[i + seq_length, [-1]]  # Next close price

    print(_x, "->", _y)
    X_data.append(_x)
    y_data.append(_y)

print(len(X_data), len(y_data))  # 725, 725

# Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_data,
    shuffle=False,
    random_state=None
)


"""
Layers
"""
# 1. RNN Layers
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)  # 왜 activation 을 썼을까
outputs, _states = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)
print(outputs.shape)  # (?, 7, 10)

# 2. FN Layers
# Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
outputs = outputs[:, -1]  # cell 의 마지막 output 만을 사용할 것임
print(outputs.shape)  # (?, 10)

W1 = tf.get_variable(name="W1", shape=[hidden_dim, output_dim],
                     initializer=tf.contrib.layers.xavier_initializer())  # (-1. 10)
b1 = tf.Variable(tf.random_normal([output_dim]), name="b1")
logits1 = tf.matmul(outputs, W1) + b1


"""
Loss & Optimizer
"""
# 여기선 0 1 의 이진 분류가 아니기 때문에 알고 있던 sequence_loss 를 사용해선 안된다.
# RMSE 를 사용하여 pred 값과 actual 값의 차이를 최소화 하는 방향으로 만들어야 한다.
loss = tf.reduce_mean(tf.square(logits1 - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


"""
Launch
"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    print("Training started ...")
    print("step | loss")
    for step in range(training_iter):
        loss_val, _ = sess.run(fetches=[loss, optimizer], feed_dict={X: X_train, Y: y_train})

        if step % 100 == 0:
            print(step, loss_val)

    print("Training finished ...")

    # Testing
    pred = sess.run(fetches=logits1, feed_dict={X: X_test})
    print(pred)
    print("Pred Size :", len(pred))


"""
Show graph
"""
plt.plot(y_test, label="Actual Value")
plt.plot(pred, label="Predicted Value")
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.legend()
plt.show()


"""
[[0.18667876 0.20948057 0.20878184 0.         0.21744815]
 [0.30697388 0.31463414 0.21899367 0.01247647 0.21698189]
 [0.21914211 0.26390721 0.2246864  0.45632338 0.22496747]
 [0.23312993 0.23641916 0.16268272 0.57017119 0.14744274]
 [0.13431201 0.15175877 0.11617252 0.39380658 0.13289962]
 [0.13973232 0.17060429 0.15860382 0.28173344 0.18171679]
 [0.18933069 0.20057799 0.19187983 0.29783096 0.2086465 ]] -> [0.14106001]
[[0.30697388 0.31463414 0.21899367 0.01247647 0.21698189]
 [0.21914211 0.26390721 0.2246864  0.45632338 0.22496747]
 [0.23312993 0.23641916 0.16268272 0.57017119 0.14744274]
 [0.13431201 0.15175877 0.11617252 0.39380658 0.13289962]
 [0.13973232 0.17060429 0.15860382 0.28173344 0.18171679]
 [0.18933069 0.20057799 0.19187983 0.29783096 0.2086465 ]
 [0.20500875 0.19950862 0.153586   0.36110962 0.14106001]] -> [0.11089532]
[[0.21914211 0.26390721 0.2246864  0.45632338 0.22496747]
 [0.23312993 0.23641916 0.16268272 0.57017119 0.14744274]
 [0.13431201 0.15175877 0.11617252 0.39380658 0.13289962]
 [0.13973232 0.17060429 0.15860382 0.28173344 0.18171679]
 [0.18933069 0.20057799 0.19187983 0.29783096 0.2086465 ]
 [0.20500875 0.19950862 0.153586   0.36110962 0.14106001]
 [0.11044525 0.12724798 0.11435324 0.35107108 0.11089532]] -> [0.11649107]

...

[[0.91021623 0.91296982 0.92617114 0.10284127 0.92046468]
 [0.91753068 0.90955899 0.93013248 0.08799857 0.92390372]
 [0.92391259 0.92282604 0.94550876 0.10049296 0.93588207]
 [0.93644323 0.93932734 0.96226395 0.10667742 0.95211558]
 [0.94518557 0.94522671 0.96376051 0.09372591 0.95564213]
 [0.9462346  0.94522671 0.97100833 0.11616922 0.9513578 ]
 [0.94789567 0.94927335 0.97250489 0.11417048 0.96645463]] -> [0.97785024]
[[0.91753068 0.90955899 0.93013248 0.08799857 0.92390372]
 [0.92391259 0.92282604 0.94550876 0.10049296 0.93588207]
 [0.93644323 0.93932734 0.96226395 0.10667742 0.95211558]
 [0.94518557 0.94522671 0.96376051 0.09372591 0.95564213]
 [0.9462346  0.94522671 0.97100833 0.11616922 0.9513578 ]
 [0.94789567 0.94927335 0.97250489 0.11417048 0.96645463]
 [0.95690035 0.95988111 0.9803545  0.14250246 0.97785024]] -> [0.98831302]
"""

"""
Training started ...
step | loss
0 1.1495243
100 0.0013377032
200 0.0012233608
300 0.001153883
400 0.0010916158
500 0.0010389442
600 0.0009957696
700 0.0009607747
800 0.0009324579
900 0.00090945006
1000 0.0008904857
1100 0.00087442045
1200 0.0008603199
1300 0.0008475334
1400 0.0008356993
1500 0.0008246919
1600 0.0008145395
1700 0.00080535037
1800 0.00079725025
1900 0.00079033436
2000 0.0007846224
2100 0.0007800311
2200 0.000776381
2300 0.0007734445
2400 0.00077099254
2500 0.00076883973
2600 0.0007668454
2700 0.0007649158
2800 0.0007629889
2900 0.00076102954
Training finished ...
"""

"""
[[0.7088326 ]
 [0.6966013 ]
 [0.6803828 ]
 [0.66432744]
 [0.6671265 ]
 [0.6921361 ]

 ...

 [0.92322314]
 [0.9284029 ]
 [0.9270874 ]
 [0.9339921 ]
 [0.9370662 ]]

Pred Size : 182
"""
