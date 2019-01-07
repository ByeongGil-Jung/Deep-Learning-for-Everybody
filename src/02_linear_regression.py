"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-06
 Time: 오후 4:10
"""

import tensorflow as tf

tf.set_random_seed(777)  # 항상 같은 seed 이기에 언제나 같은 random 값을 출력함

x_train = [1, 2, 3]
y_train = [1, 2, 3]
lr = 0.01

# tf.random_normal : normal dist 를 따르는 rand 변수 생성
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = x_train * W + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train = optimizer.minimize(cost)

# Session
sess = tf.Session()
init_op = tf.global_variables_initializer()

sess.run(init_op)

# Fitting
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print("Step : ", step)
        print("Cost : ", sess.run(cost))
        print("Weight : ", sess.run(W))
        print("Bias : ", sess.run(b))
        print()
