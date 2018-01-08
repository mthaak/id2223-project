import os
import random

import numpy as np
from skimage import io, img_as_float
from skimage.transform import resize

wikiart_dir = 'data/wikiart'
input_dir = 'data/wikiart_csv'
output_dir = 'data/npy_256x256_all'

resize_height = 272
resize_width = 272
output_height = 256
output_width = 256
output_channels = 3

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_npy_files(input_filepath, val=False):
    print(input_filepath)

    # Get data from file
    input_file = open(input_filepath, encoding='utf8')
    data = np.loadtxt(input_file, dtype=str, delimiter=',')
    n = len(data)

    splits = 10
    step = n // splits
    intervals = [(a, min(a + step, n)) for a in range(0, n, step)]

    for split, (a, b) in enumerate(intervals):
        # Create images and labels array
        images = np.empty((b - a, output_height, output_width, 3), dtype=np.float16)
        labels = np.empty(b - a, dtype=np.uint16)
        random.seed(0)  # fixed seed
        for i in range(b - a):
            image_filepath = data[a + i, 0]
            label = data[a + i, 1]

            # Prepare image
            image_read = io.imread(os.path.join(wikiart_dir, image_filepath))
            if len(image_read.shape) == 1:
                image_read = image_read[0]
            image = img_as_float(image_read)
            image = resize(image, (resize_height, resize_width))

            # Decide location of crop
            if val:
                x1, y1 = (resize_width - output_width) // 2, (resize_height - output_height) // 2
            else:
                x1 = random.randint(0, resize_width - output_width - 1)
                y1 = random.randint(0, resize_height - output_height - 1)
            x2, y2 = x1 + output_width, y1 + output_height
            cropped_image = image[y1:y2, x1:x2]

            # Flip random train images horizontally
            if not val and random.randint(0, 1):
                cropped_image = np.flip(cropped_image, axis=1)

            images[i] = cropped_image
            labels[i] = label

            print("{}/{} of {} file {}/{}".format(i + 1, b - a, "val" if val else "train", split + 1, splits))

        if output_channels == 1:
            images = np.average(images, axis=3, weights=[0.30, 0.59, 0.11])  # convert images to grayscale

        input_filename = str(os.path.basename(input_filepath).split('.')[0])

        # Save images array to npy file
        output_images_filepath = '{0}/{1}_images_{2}.npy'.format(output_dir, input_filename, str(split))
        np.save(output_images_filepath, images)

        # Save labels array to npy file
        output_labels_filepath = '{0}/{1}_labels_{2}.npy'.format(output_dir, input_filename, str(split))
        np.save(output_labels_filepath, labels)


# create_npy_files(input_dir + '/artist_train.csv')
# create_npy_files(input_dir + '/artist_val.csv')
# create_npy_files(input_dir + '/genre_train.csv')
# create_npy_files(input_dir + '/genre_val.csv')
create_npy_files(input_dir + '/style_train.csv')
create_npy_files(input_dir + '/style_val.csv', val=True)
