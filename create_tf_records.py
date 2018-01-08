import glob
import os

import numpy as np
import tensorflow as tf

input_dir = 'data/npy_256x256_all/'
output_dir = 'data/tfrecords_256x256_all/'

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecords_file(input_images_filepath, input_labels_filepath, output_filepath):
    print(input_images_filepath, input_labels_filepath)

    # Load images and labels
    images = np.load(input_images_filepath)
    labels = np.load(input_labels_filepath)
    n = len(images)

    # Initialize writer
    writer = tf.python_io.TFRecordWriter(output_filepath)

    for i in range(n):
        image = images[i]
        label = labels[i]

        # Cast from 0-1 float to 0-255 uint8
        if 'float' in image.dtype.name:
            image = (image * 256).astype(np.uint8)

        # Create feature
        feature = {
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(tf.compat.as_bytes(image.tostring()))
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write to file
        writer.write(example.SerializeToString())

        print("({}/{} of {})".format(i + 1, n, input_images_filepath))

    writer.close()


# For each pair of images and labels create a tfrecords file
for images_filepath in glob.glob(input_dir + 'style_train_images_*') + glob.glob(input_dir + 'style_val_images_*'):
    labels_filepath = images_filepath.replace('images', 'labels')
    if os.path.isfile(labels_filepath):
        output_filename = os.path.basename(images_filepath).replace('images_', '').replace('.npy', '.tfrecords')
        create_tfrecords_file(images_filepath, labels_filepath, os.path.join(output_dir, output_filename))
