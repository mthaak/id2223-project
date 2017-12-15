# from __future__ import division
# from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import dataset

FLAGS = None


# Based on https://gist.github.com/ambodi/408301bc5bc07bc5afa8748513ab9477

def main(_):
    """Run the NN."""
    mnist = dataset.read_data_sets(FLAGS.class_type, one_hot=True)

    # Set NN parameters
    image_height = 227
    image_width = 227
    image_channels = 3
    num_classes = {'artist': 23, 'genre': 10, 'style': 27}[FLAGS.class_type]
    num_iterations = 1000
    batch_size = 100
    learning_rate = 0.5

    # Define placeholders and variables
    flat = image_height * image_width * image_channels
    x = tf.placeholder(tf.float32, [None, image_height, image_width, image_channels])
    xx = tf.reshape(x, [-1, flat])
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    w = tf.Variable(tf.zeros([flat, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    # Define model
    y = tf.matmul(xx, w) + b

    # Define loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    # The raw formulation of cross-entropy,
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
    # can be numerically unstable.
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.

    # Define optimizer
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # Define accuracy formula
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Start session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Train
        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if i % 100 == 99:
                # Test trained model
                acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                print('Iter {0}: {1}'.format(i + 1, acc))


if __name__ == '__main__':
    # Run NN using arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_type', type=str, default='style', help='The class type on which to train and test')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
