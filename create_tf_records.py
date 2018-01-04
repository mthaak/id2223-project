import numpy as np
import tensorflow as tf

input_dir = 'data/npy_256x256/'
output_dir = 'data/tfrecords_256x256/'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecords_file(input_images_filepath, input_labels_filepath, output_filepath):
    # Load images and labels
    images = np.load(input_images_filepath)
    labels = np.load(input_labels_filepath)
    n = len(images)

    # Initialize writer
    writer = tf.python_io.TFRecordWriter(output_filepath)
    for i in range(n):
        image = images[i]
        label = labels[i]

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

        print("({}/{})".format(i + 1, n))

    writer.close()


# create_tfrecords_file(input_dir + 'style_train_images.npy', input_dir + 'style_train_labels.npy',
#                       output_dir + 'style_train.tfrecords')
create_tfrecords_file(input_dir + 'style_val_images.npy', input_dir + 'style_val_labels.npy',
                      output_dir + 'style_val.tfrecords')
