"""
 Created by IntelliJ IDEA.
 Project: Deep-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-01-09
 Time: 오후 9:04
"""

import tensorflow as tf

tf.set_random_seed(777)

# Parameters
lr = 1e-5

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name="weight1")
w2 = tf.Variable(tf.random_normal([1]), name="weight2")
w3 = tf.Variable(tf.random_normal([1]), name="weight3")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Making Model
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

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
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})

    if step % 100 == 0:
        print(step, cost_val, hy_val)


"""
step | cost | pred
0 62547.29 [-75.96345  -78.27629  -83.83015  -90.80436  -56.976482]
100 13.247165 [146.06377 188.31447 178.98654 195.37894 146.31259]
200 12.5621395 [146.20279 188.21931 179.0293  195.4085  146.1889 ]
300 11.913301 [146.3381  188.12671 179.07095 195.43723 146.06857]
400 11.298648 [146.46979 188.03654 179.11147 195.46513 145.95143]
500 10.716393 [146.59799 187.9488  179.15092 195.4923  145.83746]
600 10.164846 [146.72278 187.8634  179.18935 195.51869 145.72656]
700 9.642378 [146.84425 187.78029 179.22675 195.54437 145.61865]
800 9.14744 [146.96246 187.69936 179.26315 195.5693  145.51361]
900 8.678604 [147.07758 187.6206  179.29858 195.59357 145.41145]
1000 8.234492 [147.18959 187.54395 179.33308 195.61716 145.31201]
1100 7.8137712 [147.29865 187.46933 179.36667 195.64008 145.21526]
1200 7.4152174 [147.4048  187.3967  179.39938 195.66235 145.12111]
1300 7.0377045 [147.50812 187.326   179.4312  195.68402 145.0295 ]
1400 6.680027 [147.6087  187.25717 179.46219 195.70505 144.94034]
1500 6.341264 [147.7066  187.1902  179.49237 195.72554 144.8536 ]
1600 6.020313 [147.80191 187.125   179.52174 195.74542 144.7692 ]
1700 5.716288 [147.89468 187.06155 179.55034 195.76476 144.68706]
1800 5.4283 [147.98499 186.99979 179.57817 195.78354 144.60715]
1900 5.1554766 [148.07289 186.93965 179.6053  195.8018  144.5294 ]
2000 4.8970113 [148.15845 186.8811  179.63167 195.81953 144.45372]
"""
