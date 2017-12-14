import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

ITERATIONS = 1000

# 1. Define Variables and placeholders
pkeep = tf.placeholder(tf.float32, name='pkeep')

global_step = tf.Variable(0, trainable=False)
STARTER_LEARNING_RATE = 0.01
LEARNING_RATE = tf.train.exponential_decay(STARTER_LEARNING_RATE, global_step, 100000, 0.96, staircase=True)

# 1. Read fashion MNIST dataset
# mnist = input_data.read_data_sets('data/mnist', one_hot=True)
mnist = input_data.read_data_sets('data/fashion', one_hot=True)

# 2. Define variables and placeholders
X = tf.placeholder(tf.float32, [None, 784], name='X')  # the first dimension (None) will index the images
Y_ = tf.placeholder(tf.float32, [None, 10], name='Y_')  # correct answers
# Weights initialised with small random values between -0.2 and +0.2
C1_W = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1))
C1_B = tf.Variable(tf.zeros([4]))
C2_W = tf.Variable(tf.truncated_normal([5, 5, 4, 8], stddev=0.1))
C2_B = tf.Variable(tf.zeros([8]))
C3_W = tf.Variable(tf.truncated_normal([4, 4, 8, 12], stddev=0.1))
C3_B = tf.Variable(tf.zeros([12]))
F1_W = tf.Variable(tf.truncated_normal([7 * 7 * 12, 200], stddev=0.1))
F1_B = tf.Variable(tf.zeros([200]))
F2_W = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
F2_B = tf.Variable(tf.zeros([10]))

# 3. Define model
# Reshape
XX = tf.reshape(X, [-1, 28, 28, 1])
# Conv layer 1
C1 = tf.nn.relu(tf.nn.conv2d(XX, C1_W, strides=[1, 1, 1, 1], padding='SAME') + C1_B)
# Conv layer 2
C2 = tf.nn.relu(tf.nn.conv2d(C1, C2_W, strides=[1, 2, 2, 1], padding='SAME') + C2_B)
# Conv layer 3
C3 = tf.nn.relu(tf.nn.conv2d(C2, C3_W, strides=[1, 2, 2, 1], padding='SAME') + C3_B)
# Fully connected layer 1
FC1_flat = tf.reshape(C3, [-1, 7*7*12])
FC1 = tf.nn.relu(tf.matmul(FC1_flat, F1_W) + F1_B)
# Drop out
DROP = tf.nn.dropout(FC1, pkeep)
# Fully connected layer 2
FC2 = tf.matmul(DROP, F2_W) + F2_B

Y = FC2

# 4. Define loss function (with cross-entropy on logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_)
loss_function = tf.reduce_mean(cross_entropy)

# 5. Define accuracy formula
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 6. Initialize optimizer
# optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_function, global_step=global_step)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_function, global_step=global_step)


# 7. Start session and run optimizer
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    costs = []
    accuracies = []
    for i in range(ITERATIONS):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        a, cost = sess.run([optimizer, loss_function], feed_dict={X: batch_xs, Y_: batch_ys, pkeep: 0.95})
        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1})
            costs.append(cost)
            accuracies.append(acc)
            print(i, acc, cost)

print("Final accuracy: {0}".format(accuracies[-1]))

# 8. Plot both costs and accuracies for each epoch
plt.figure(figsize=(16, 5))
plt.plot(np.squeeze(costs), color='red')
plt.xlim(0, ITERATIONS / 100)
plt.ylabel("cost")
plt.xlabel("iterations")
plt.title("learning rate = {rate}".format(rate=LEARNING_RATE))

plt.figure(figsize=(16, 5))
plt.plot(np.squeeze(accuracies), color='blue')
plt.xlim(0, ITERATIONS / 100)
plt.ylabel("accuracy")
plt.xlabel("iterations")
plt.title("learning rate = {rate}".format(rate=LEARNING_RATE))
plt.show()
