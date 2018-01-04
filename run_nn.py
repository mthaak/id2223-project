import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import dataset

FLAGS = None

# Set NN parameters
image_height = 16
image_width = 16
image_channels = 3
num_classes = 27
num_iterations = 10000
batch_size = 100
learning_rate = 0.0001
pkeep = tf.placeholder(tf.float32)
drop = 0.05
plot_step = 10


def setup_nn(x):
    # Define placeholders and variables
    # flat = image_height * image_width * image_channels
    flat = 256
    w = tf.Variable(tf.zeros([flat, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    # Define model
    xx = tf.reshape(x, [-1, flat])
    y = tf.matmul(xx, w) + b

    return y


def setup_nn2(x):
    # Define model
    # flat = image_height * image_width * image_channels
    flat = 4096
    xx = tf.reshape(x, [-1, flat])
    y1 = tf.layers.dense(xx, units=flat, activation=tf.nn.relu)
    y2 = tf.layers.dense(y1, units=flat / 2, activation=tf.nn.relu)
    y3 = tf.layers.dense(y2, units=27, activation=tf.nn.relu)

    return y3


def setup_cnn(x):
    # Define variables and placeholders
    # Weights initialised with small random values between -0.2 and +0.2
    c1_w = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1))
    c1_b = tf.Variable(tf.zeros([4]))
    c2_w = tf.Variable(tf.truncated_normal([5, 5, 4, 8], stddev=0.1))
    c2_b = tf.Variable(tf.zeros([8]))
    c3_w = tf.Variable(tf.truncated_normal([4, 4, 8, 12], stddev=0.1))
    c3_b = tf.Variable(tf.zeros([12]))
    f1_w = tf.Variable(tf.truncated_normal([8 * 8 * 12, 200], stddev=0.1))
    f1_b = tf.Variable(tf.zeros([200]))
    f2_w = tf.Variable(tf.truncated_normal([200, num_classes], stddev=0.1))
    f2_b = tf.Variable(tf.zeros([num_classes]))

    # Define model
    # Conv layer 1
    c1 = tf.nn.relu(tf.nn.conv2d(x, c1_w, strides=[1, 1, 1, 1], padding='SAME') + c1_b)
    # Conv layer 2
    c2 = tf.nn.relu(tf.nn.conv2d(c1, c2_w, strides=[1, 2, 2, 1], padding='SAME') + c2_b)
    # Conv layer 3
    c3 = tf.nn.relu(tf.nn.conv2d(c2, c3_w, strides=[1, 2, 2, 1], padding='SAME') + c3_b)
    # Fully connected layer 1
    fc1_flat = tf.reshape(c3, [-1, 8 * 8 * 12])
    fc1 = tf.nn.relu(tf.matmul(fc1_flat, f1_w) + f1_b)
    # Drop out
    drop = tf.nn.dropout(fc1, pkeep)
    # Fully connected layer 2
    fc2 = tf.matmul(drop, f2_w) + f2_b

    y = fc2

    return y


def setup_cnn2(x):
    # Convolutional layer #1 and pooling layer #1
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional layer #2 and pooling layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=drop, training=True)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=num_classes)
    y = logits

    return y


def plot(accuracies, costs):
    # 8. Plot both costs and accuracies for each epoch
    plt.figure(figsize=(16, 5))
    plt.plot(np.arange(0, num_iterations + 1, plot_step), np.squeeze(accuracies), color='blue')
    plt.xlim(0, num_iterations)
    plt.ylabel("accuracy")
    plt.xlabel("iterations")
    plt.title("learning rate = {rate}".format(rate=learning_rate))

    plt.figure(figsize=(16, 5))
    plt.plot(np.arange(0, num_iterations + 1, plot_step), np.squeeze(costs), color='red')
    plt.xlim(0, num_iterations + 1)
    plt.ylabel("cost")
    plt.xlabel("iterations")
    plt.title("learning rate = {rate}".format(rate=learning_rate))

    plt.show()


# Based on https://gist.github.com/ambodi/408301bc5bc07bc5afa8748513ab9477

def main(_):
    """Run the NN."""
    paintings = dataset.read_data_sets(FLAGS.class_type, one_hot=True)

    # Set up neural network
    # x = tf.placeholder(tf.float32, [None, image_height, image_width, image_channels])
    x = tf.placeholder(tf.float32, [None, 4096])
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    y = setup_nn2(x)

    # Define loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss_function = tf.reduce_mean(cross_entropy)
    # The raw formulation of cross-entropy,
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
    # can be numerically unstable.
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.

    # Define optimizer
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

    # Define accuracy formula
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracies = []
    costs = []

    # Start session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        test_images = paintings.test.images[range(0, len(paintings.test.images), 10)]
        test_labels = paintings.test.labels[range(0, len(paintings.test.labels), 10)]

        # Train
        for i in range(num_iterations + 1):
            batch_xs, batch_ys = paintings.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, pkeep: 1 - drop})

            if i % plot_step == 0:
                # Test trained model
                acc, cost = sess.run([accuracy, loss_function], feed_dict={x: test_images, y_: test_labels, pkeep: 1.0})
                accuracies.append(acc)
                costs.append(cost)
                print('Iter {0}: {1} {2}'.format(i + 1, acc, cost))

    plot(accuracies, costs)


if __name__ == '__main__':
    # Run NN using arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_type', type=str, default='style', help='The class type on which to train and test')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
