import os
import random

import numpy as np
from skimage import io, img_as_float
from skimage.transform import resize

wikiart_dir = 'data/wikiart'
input_dir = 'data/wikiart_sample_csv'
output_dir = 'data/npy_256x256'

resize_height = 272
resize_width = 272
output_height = 256
output_width = 256
output_channels = 3
output_num = 1

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_npy_files(input_filepath, val=False):
    print(input_filepath)

    # Get data from file
    input_file = open(input_filepath, encoding='utf8')
    data = np.loadtxt(input_file, dtype=str, delimiter=',')
    n = len(data)

    # Create images and labels array
    images = np.empty((n * output_num, output_height, output_width, 3), dtype=np.float16)
    labels = np.empty(n * output_num, dtype=np.uint16)
    random.seed(0)  # fixed seed
    for i, image_file in enumerate(data[:, 0]):
        image = img_as_float(io.imread(wikiart_dir + '/' + image_file))
        image = resize(image, (resize_height, resize_width))
        for j in range(output_num):
            x1 = random.randint(0, resize_width - output_width - 1)
            y1 = random.randint(0, resize_height - output_height - 1)
            x2, y2 = x1 + output_width, y1 + output_height
            cropped_image = image[y1:y2, x1:x2]
            if not val and random.randint(0, 1):
                cropped_image = np.flip(cropped_image, axis=1)  # flip random train images horizontally
            images[i * output_num + j] = cropped_image
            labels[i * output_num + j] = data[i, 1]

        print("({}/{})".format(i + 1, n))

    if output_channels == 1:
        images = np.average(images, axis=3, weights=[0.30, 0.59, 0.11])  # convert images to grayscale

    input_filename = str(os.path.basename(input_filepath).split('.')[0])

    # Save images array to npy file
    output_images_filepath = output_dir + '/' + input_filename + '_images.npy'
    np.save(output_images_filepath, images)

    # Save labels array to npy file
    output_labels_filepath = output_dir + '/' + input_filename + '_labels.npy'
    np.save(output_labels_filepath, labels)


# create_npy_files(input_dir + '/artist_train.csv')
# create_npy_files(input_dir + '/artist_val.csv')
# create_npy_files(input_dir + '/genre_train.csv')
# create_npy_files(input_dir + '/genre_val.csv')
create_npy_files(input_dir + '/style_train.csv')
create_npy_files(input_dir + '/style_val.csv', val=True)
