import os

import numpy as np
# from scipy import io
from skimage import io
from skimage import transform

wikiart_dir = 'data/wikiart'
input_dir = 'data/wikiart_sample_csv'
output_dir = 'data/npy'

height = 227
width = 227
channels = 3

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_npy(input_filepath, output_dir):
    data = np.loadtxt(input_filepath, dtype=str, delimiter=',')
    n = len(data)

    s = 50
    n = s

    # Create images array
    images = np.empty((n, height, width, channels), dtype=np.float16)
    for i, image_file in enumerate(data[:n, 0]):
        image = io.imread(wikiart_dir + '/' + image_file)
        resized_image = transform.resize(image, (height, width))
        images[i] = resized_image

        print("({0}/{1})".format(i + 1, n))

    # Create labels array
    labels = data[:, 1]

    input_filename = os.path.basename(input_filepath).split('.')[0]

    # Save images array to npy file
    output_images_filepath = output_dir + '/' + input_filename + '_images.npy'
    np.save(output_images_filepath, images)

    # Save labels array to npy file
    output_labels_filepath = output_dir + '/' + input_filename + '_labels.npy'
    np.save(output_labels_filepath, labels)


# create_npy(input_dir + '/artist_train.csv', output_dir)
create_npy(input_dir + '/artist_val.csv', output_dir)
# create_npy(input_dir + '/genre_train.csv', output_dir)
# create_npy(input_dir + '/genre_val.csv', output_dir)
# create_npy(input_dir + '/style_train.csv', output_dir)
# create_npy(input_dir + '/style_val.csv', output_dir)
