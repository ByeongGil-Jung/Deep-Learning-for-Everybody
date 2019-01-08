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


"""
Step :  0
Cost :  2.823292
Weight :  [2.1286771]
Bias :  [-0.8523567]

Step :  20
Cost :  0.19035067
Weight :  [1.533928]
Bias :  [-1.0505961]

Step :  40
Cost :  0.15135698
Weight :  [1.4572546]
Bias :  [-1.0239124]

Step :  60
Cost :  0.1372696
Weight :  [1.4308538]
Bias :  [-0.9779527]

...

Step :  1960
Cost :  1.4639714e-05
Weight :  [1.004444]
Bias :  [-0.01010205]

Step :  1980
Cost :  1.3296165e-05
Weight :  [1.0042351]
Bias :  [-0.00962736]

Step :  2000
Cost :  1.20760815e-05
Weight :  [1.0040361]
Bias :  [-0.00917497]
"""
