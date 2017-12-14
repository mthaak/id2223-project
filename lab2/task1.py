import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

ITERATIONS = 10000
LEARNING_RATE = 0.005

# 1. Read fashion MNIST dataset
# mnist = input_data.read_data_sets('data/mnist', one_hot=True)
mnist = input_data.read_data_sets('data/fashion', one_hot=True)

# 2. Define variables and placeholders
X = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
XX = tf.reshape(X, [-1, 784])

# 3. Define model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# 4. Define loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))

# 5. Define accuracy formula
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 6. Initialize optimizer
# optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# 7. Start session and run optimizer
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    costs = []
    accuracies = []
    for i in range(ITERATIONS):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        a, cost = sess.run([optimizer, cross_entropy], feed_dict={X: batch_xs, Y_: batch_ys})
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


