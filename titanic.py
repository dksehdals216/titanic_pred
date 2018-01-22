

import numpy as np
import tensorflow as tf

#import test and train data
#x_data is 792x14, while y is 792_1
inp_train = np.loadtxt('train_data.csv',delimiter=',', dtype=np.float32)
inp_test = np.loadtxt('test_data.csv',delimiter=',', dtype=np.float32)
x_data = inp_train[:,3:] 
y_data = inp_train[:, [2]] 

x = tf.placeholder(tf.float32, shape=[None, 14])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([14,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#accuracy computation
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x: x_data, y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
