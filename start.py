# Thanks to https://gist.github.com/eerwitt/518b0c9564e500b4b50f

# Typical setup to include TensorFlow.
import glob

import tensorflow as tf

###################

# # Make a queue of file names including all the JPEG images files in the relative
# # image directory.
# filenames = glob.glob("data/*.jpg")
# filename_queue = tf.train.string_input_producer(filenames)
#
# # Read an entire image file which is required since they're JPEGs, if the images
# # are too large they could be split in advance to smaller files or use the Fixed
# # reader to split up the file.
# image_reader = tf.WholeFileReader()
#
# # Read a whole file from the queue, the first returned value in the tuple is the
# # filename which we are ignoring.
# _, image_file = image_reader.read(filename_queue)
#
# # Decode the image as a JPEG file, this will turn it into a Tensor which we can
# # then use in training.
# image = tf.image.decode_jpeg(image_file)

###################

def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line.split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label


# Reads pfathes of images together with their labels
image_list, label_list = read_labeled_image_list("data/labels.txt")

images = tf.convert_to_tensor(image_list, dtype=tf.string)
labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

num_epochs = 1
# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels],
                                            num_epochs=num_epochs,
                                            shuffle=True)

image, label = read_images_from_disk(input_queue)

#####################

# Resize image
image = tf.image.resize_images(image, [227, 227])
# resized_image = tf.image.resize_images(image, [227, 227])

# Define variables and placeholders
X = tf.placeholder(tf.float32, [None, 227, 277])
Y_ = tf.placeholder(tf.float32, [None, 2])
W = tf.Variable(tf.zeros([227, 277, 2]))
b = tf.Variable(tf.zeros([2]))
# XX = tf.reshape(X, [-1, 784])

# Define model
Y = tf.nn.softmax(tf.matmul(X, W) + b)

# Define loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))

# Define accuracy formula
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize optimizer
optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)


# optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

def shape(tensor):
    try:
        return [len(tensor)] + shape(tensor[0])
    except TypeError:
        return []


# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    # tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()

    # Coordinate the loading of image files.
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    # image_tensor = sess.run([resized_image])
    # print("shape:", shape(image_tensor))
    # print(image_tensor)

    # Finish off the filename queue coordinator.
    # coord.request_stop()
    # coord.join(threads)

    batch_size = 1

    for i in range(5):
        image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size)
        image_feed, label_feed = sess.run([image_batch, label_batch])
        a, cost = sess.run([optimizer, cross_entropy], feed_dict={X: image_feed, Y_: label_feed})
        acc = sess.run(accuracy, feed_dict={X: image, Y_: label})
        print(i, acc, cost)
