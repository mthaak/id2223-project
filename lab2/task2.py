import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

ITERATIONS = 1000
LEARNING_RATE = 0.01

# 1. Read fashion MNIST dataset
# mnist = input_data.read_data_sets('data/mnist', one_hot=True)
mnist = input_data.read_data_sets('data/fashion', one_hot=True)

# 2. Define variables and placeholders
X = tf.placeholder(tf.float32, [None, 784])  # the first dimension (None) will index the images
Y_ = tf.placeholder(tf.float32, [None, 10])  # correct answers
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
# Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
# Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
# Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
# Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.nn.softmax(tf.matmul(Y4, W5) + B5)
Y = Ylogits

# 4. Define loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,
                                                        labels=Y_)  # calculate cross-entropy with logits
loss_function = tf.reduce_mean(cross_entropy)

# 5. Define accuracy formula
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 6. Initialize optimizer
# optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_function)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_function)

# 7. Start session and run optimizer
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    costs = []
    accuracies = []
    for i in range(ITERATIONS):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        a, cost = sess.run([optimizer, loss_function], feed_dict={X: batch_xs, Y_: batch_ys})
        acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        costs.append(cost)
        accuracies.append(acc)
        print(i, acc, cost)

print("Final loss: {0}".format(costs[-1]))
print("Final accuracy: {0}".format(accuracies[-1]))

# 8. Plot both costs and accuracies for each epoch
plt.figure(figsize=(16, 5))
plt.plot(np.squeeze(costs), color='red')
plt.xlim(0, ITERATIONS)
plt.ylabel("cost")
plt.xlabel("iterations")
plt.title("learning rate = {rate}".format(rate=LEARNING_RATE))

plt.figure(figsize=(16, 5))
plt.plot(np.squeeze(accuracies), color='blue')
plt.xlim(0, ITERATIONS)
plt.ylabel("accuracy")
plt.xlabel("iterations")
plt.title("learning rate = {rate}".format(rate=LEARNING_RATE))
plt.show()
