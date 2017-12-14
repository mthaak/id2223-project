# coding: utf-8

# In[43]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

ITERATIONS = 10000
# 1. Define Variables and placeholders
pkeep = tf.placeholder(tf.float32, name='pkeep')

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
LEARNING_RATE = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)

# 1. Read fashion MNIST dataset
# mnist = input_data.read_data_sets('data/mnist', one_hot=True)
mnist = input_data.read_data_sets('data/fashion', one_hot=True)

# 2. Define variables and placeholders
X = tf.placeholder(tf.float32, [None, 784], name='X')  # the first dimension (None) will index the images
Y_ = tf.placeholder(tf.float32, [None, 10], name='Y_')  # correct answers
# Weights initialised with small random values between -0.2 and +0.2
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
B1 = tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
B2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
B3 = tf.Variable(tf.zeros([60]))
W4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
B4 = tf.Variable(tf.zeros([30]))
W5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# 3. Define model
XX = tf.reshape(X, [-1, 784])
# 2. Define the model
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
Y1d= tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.sigmoid(tf.matmul(Y1d, W2) + B2)
Y2d= tf.nn.dropout(Y2, pkeep)
Y3 = tf.nn.sigmoid(tf.matmul(Y2d, W3) + B3)
Y3d= tf.nn.dropout(Y3, pkeep)
Y4 = tf.nn.sigmoid(tf.matmul(Y3d, W4) + B4)
Y4d= tf.nn.dropout(Y4, pkeep)

# Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
# Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
# Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
# Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.nn.softmax(tf.matmul(Y4d, W5) + B5)
Y = Ylogits

# 4. Define loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,
                                                        labels=Y_)  # calculate cross-entropy with logits
loss_function = tf.reduce_mean(cross_entropy)

# 5. Define accuracy formula
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 6. Initialize optimizer
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_function, global_step=global_step)
# optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_function)


# 7. Start session and run optimizer
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    costs = []
    accuracies = []
    for i in range(ITERATIONS):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        a, cost = sess.run([optimizer, loss_function], feed_dict={X: batch_xs, Y_: batch_ys, pkeep:1})
        if i %100 ==0:
            acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep:0.95})
            costs.append(cost)
            accuracies.append(acc)
            print(i, acc, cost)

print("Final accuracy: {0}".format(accuracies[-1]))

# 8. Plot both costs and accuracies for each epoch
plt.figure(figsize=(16, 5))
plt.plot(np.squeeze(costs), color='red')
plt.xlim(0, ITERATIONS/100)
plt.ylabel("cost")
plt.xlabel("iterations")
plt.title("learning rate = {rate}".format(rate=LEARNING_RATE))

plt.figure(figsize=(16, 5))
plt.plot(np.squeeze(accuracies), color='blue')
plt.xlim(0, ITERATIONS/100)
plt.ylabel("accuracy")
plt.xlabel("iterations")
plt.title("learning rate = {rate}".format(rate=LEARNING_RATE))
plt.show()

# TODO
# Questions
# 1. What is the maximum accuracy that you can get in each setting for running your model with 10000 iterations?
# 2. Is there a big difference between the convergence rate of the sigmoid and the ReLU? If yes, what is the reason for the difference?
# 3. What is the reason that we use the softmax in our output layer?
# 4. By zooming into the second half of the epochs in accuracy and loss plot, do you see any strange behaviour? What is the reason and how you can overcome them? (e.g., look at fluctuations or sudden loss increase after a period of decreasing loss).
