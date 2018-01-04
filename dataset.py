"""A generic module to read data."""
import collections

import numpy


# Based on  on https://gist.github.com/ambodi/408301bc5bc07bc5afa8748513ab9477

class DataSet(object):
    """Dataset class object."""

    def __init__(self, images, labels):
        """Initialize the class."""
        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


def read_data_sets(class_, one_hot=False):
    train_images = numpy.load('data/color2/{0}_train_images.npy'.format(class_))
    train_labels = numpy.load('data/color2/{0}_train_labels.npy'.format(class_))
    test_images = numpy.load('data/color2/{0}_val_images.npy'.format(class_))
    test_labels = numpy.load('data/color2/{0}_val_labels.npy'.format(class_))

    if one_hot:
        num_classes = {'artist': 23, 'genre': 10, 'style': 27}[class_]
        train_labels = dense_to_one_hot(train_labels, num_classes)
        test_labels = dense_to_one_hot(test_labels, num_classes)

    train = DataSet(train_images, train_labels)
    test = DataSet(test_images, test_labels)
    ds = collections.namedtuple('Datasets', ['train', 'test'])

    return ds(train=train, test=test)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot
